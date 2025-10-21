# /***************************************************************************************************
# * Copyright (c) 2022 Iluvatar CoreX. All rights reserved.
# * Copyright Declaration: This software, including all of its code and documentation,
# * except for the third-party software it contains, is a copyrighted work of Shanghai Iluvatar CoreX
# * Semiconductor Co., Ltd. and its affiliates ("Iluvatar CoreX") in accordance with the PRC Copyright
# * Law and relevant international treaties, and all rights contained therein are enjoyed by Iluvatar
# * CoreX. No user of this software shall have any right, ownership or interest in this software and
# * any use of this software shall be in compliance with the terms and conditions of the End User
# * License Agreement.
#  **************************************************************************************************/

import os
from typing import Tuple

import torch
import torch.distributed as dist

import amp_C
import apex_C
from apex import amp
from apex.amp import _amp_state
from apex.contrib.optimizers.distributed_fused_lamb import DistributedFusedLAMB
from apex.optimizers import FusedLAMB
from apex.parallel import DistributedDataParallel as APEX_DDP
from apex.parallel.distributed import flat_dist_call
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as NativeDDP
from torch.optim import Optimizer

import utils

from train.event.base import BaseTrainingEventInterface
from train.event.base import BatchType, BERT_MODEL

from converter import convert_model
from distributed_fused_lamb import _pipeline_block_reductions_patched, _pipeline_step_patched


class ApexTrainingEvent(BaseTrainingEventInterface):

    def __init__(self, config):
        super(ApexTrainingEvent, self).__init__(config)
        self.num_iters_per_dataloader = 1

        self.optimizer = None

        self.overflow_buf = None
        if config.allreduce_post_accumulation:
            self.overflow_buf = torch.cuda.IntTensor([0])

    def on_init_start(self):
        if self.config.unpad:
            import ext_ops
            torch.cuda.synchronize()
            ext_ops.InitMHACUDAExtension()
            torch.cuda.synchronize()
            print("Init unpad")

    def convert_model(self, model: BERT_MODEL) -> BERT_MODEL:
        return convert_model(model, self.config)

    def create_optimizer(self, model: BERT_MODEL) -> Optimizer:
        config = self.config
        param_optimizer = list(model.named_parameters())

        no_decay = ['bias', 'gamma', 'beta', 'LayerNorm']

        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': self.config.weight_decay_rate},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

        if config.distributed_lamb:
            DistributedFusedLAMB._pipeline_block_reductions = _pipeline_block_reductions_patched
            DistributedFusedLAMB._pipeline_step = _pipeline_step_patched
            optimizer = DistributedFusedLAMB(optimizer_grouped_parameters, lr=config.learning_rate,
                                             betas=(config.opt_lamb_beta_1, config.opt_lamb_beta_2),
                                             eps=1e-6,
                                             max_grad_norm=1.0,
                                             overlap_reductions=config.dwu_overlap_reductions,
                                             dwu_group_size=config.dwu_group_size,
                                             dwu_num_blocks=config.dwu_num_blocks,
                                             dwu_num_chunks=config.dwu_num_chunks,
                                             dwu_num_rs_pg=config.dwu_num_rs_pg,
                                             dwu_num_ar_pg=config.dwu_num_ar_pg,
                                             dwu_num_ag_pg=config.dwu_num_ag_pg,
                                             use_nvlamb=False,
                                             e5m2_allgather=config.dwu_e5m2_allgather)
            optimizer.set_global_scale(float(os.getenv("INIT_LOSS_SCALE", 2 ** 20)))
        else:
            optimizer = FusedLAMB(optimizer_grouped_parameters,
                                  lr=config.learning_rate,
                                  betas=(config.opt_lamb_beta_1, config.opt_lamb_beta_2))

        return optimizer

    def model_to_fp16(self, model: BERT_MODEL, optimizer: Optimizer) -> Tuple[BERT_MODEL, Optimizer]:
        self.optimizer = optimizer
        config = self.config
        if config.fp16 and config.bypass_amp:
            model.half()

        if config.fp16 and not config.bypass_amp:
            if config.distributed_lamb:
                model.half()
            elif config.fp16:
                if config.loss_scale == 0:
                    if config.opt_level == 'O0':
                        loss_scale = '1.0'
                        master_weights = False
                    elif config.opt_level == 'O1':
                        loss_scale = 'dynamic'
                        master_weights = None
                    else:
                        loss_scale = 'dynamic'
                        master_weights = True
                    model, optimizer = amp.initialize(model, optimizer, opt_level=config.opt_level,
                                                      loss_scale=loss_scale,
                                                      master_weights=master_weights)
                else:
                    # assert False, "code path not tested with cuda graphs"
                    model, optimizer = amp.initialize(model, optimizer, opt_level=config.opt_level,
                                                      loss_scale=config.loss_scale)
                amp._amp_state.loss_scalers[0]._loss_scale = float(os.getenv("INIT_LOSS_SCALE", 2 ** 20))
        return model, optimizer

    def model_to_ddp(self, model: BERT_MODEL) -> BERT_MODEL:
        config = self.config
        use_ddp = dist.is_initialized()
        if use_ddp and not config.distributed_lamb and not config.allreduce_post_accumulation:
            if config.ddp_type == 'native':
                model = NativeDDP(model,
                                  device_ids=[config.local_rank],
                                  bucket_cap_mb=100,
                                  gradient_as_bucket_view=config.use_gradient_as_bucket_view)
            elif config.ddp_type == 'apex':
                model = APEX_DDP(model,
                                 message_size=250000000,
                                 delay_allreduce=True,
                                 gradient_predivide_factor=torch.distributed.get_world_size())
            else:
                assert False, "Invalid DDP type"

        if use_ddp and config.distributed_lamb:
            flat_dist_call([param.data for param in model.parameters()], torch.distributed.broadcast, (0,))

        return model

    def on_step_begin(self, step: int):
        update_step = step % self.config.gradient_accumulation_steps == 0
        if self.config.distributed_lamb:
            # TODO: 仅适用于8卡，需针对该参数调整
            num_iters = 2835
            self.optimizer.set_last_step((step % num_iters) == 0)
            self.optimizer.set_is_accumulation_step(not update_step)

    def on_step_end(self, step: int):
        if self.config.distributed_lamb:
            self.optimizer._lr = torch.tensor([self.optimizer.param_groups[0]['lr']]).to(self.config.device)

    def on_backward(self, step: int, loss: torch.Tensor, optimizer: Optimizer, grad_scaler: GradScaler=None):

        loss = loss / self.config.gradient_accumulation_steps

        if self.config.bypass_amp:
            loss.backward()
        elif self.config.distributed_lamb:
            optimizer._lazy_init_stage1()
            grad_scaler.scale(loss).backward()
            optimizer._lazy_init_stage2()
        else:
            with amp.scale_loss(loss, optimizer, delay_overflow_check=self.config.allreduce_post_accumulation) as scaled_loss:
                scaled_loss.backward()

        update_step = step % self.config.gradient_accumulation_steps == 0
        if update_step:
            self.update_model_params(loss, optimizer, grad_scaler)

    def update_model_params(self, loss, optimizer: Optimizer, grad_scaler: GradScaler=None):
        config = self.config
        if config.allreduce_post_accumulation and config.use_cuda_graph:
            assert False, "code path not tested with cuda graphs"
        if config.distributed_lamb:
            optimizer.set_global_scale(grad_scaler._get_scale_async())
            optimizer.complete_reductions()
            grad_scaler.step(optimizer)
            grad_scaler.update()

            found_inf = optimizer._overflow_buf  # GPU tensor

        elif config.allreduce_post_accumulation:
            # manually allreduce gradients after all accumulation steps
            # check for Inf/NaN
            # 1. allocate an uninitialized buffer for flattened gradient
            # torch.nn.utils.clip_grad_norm_(parameters=amp.master_params(optimizer), max_norm=1.0, norm_type=2.0)
            scaler = _amp_state.loss_scalers[0]
            master_grads = [p.grad for p in amp.master_params(optimizer) if p.grad is not None]
            flat_grad_size = sum(p.numel() for p in master_grads)
            allreduce_dtype = torch.float16 if config.allreduce_post_accumulation_fp16 else torch.float32
            flat_raw = torch.empty(flat_grad_size, device='cuda', dtype=allreduce_dtype)
            # 2. combine unflattening and predivision of unscaled 'raw' gradient
            allreduced_views = apex_C.unflatten(flat_raw, master_grads)
            self.overflow_buf.zero_()
            amp_C.multi_tensor_scale(65536,
                                     self.overflow_buf,
                                     [master_grads, allreduced_views],
                                     scaler.loss_scale() / (
                                             torch.distributed.get_world_size() * config.gradient_accumulation_steps))
            # 3. sum gradient across ranks. Because of the predivision, this averages the gradient
            torch.distributed.all_reduce(flat_raw)
            # 4. combine unscaling and unflattening of allreduced gradient
            self.overflow_buf.zero_()
            amp_C.multi_tensor_scale(65536,
                                     self.overflow_buf,
                                     [allreduced_views, master_grads],
                                     1. / scaler.loss_scale())
            # 5. update loss scale
            scaler = _amp_state.loss_scalers[0]
            old_overflow_buf = scaler._overflow_buf
            scaler._overflow_buf = self.overflow_buf
            had_overflow = scaler.update_scale()
            scaler._overflow_buf = old_overflow_buf
            # 6. call optimizer step function
            if had_overflow == 0:
                optimizer.step()
            else:
                # Overflow detected, print message and clear gradients
                if utils.is_main_process():
                    print("Overflow detected, reduced loss_scaler to %f" % (scaler.loss_scale()))
                if _amp_state.opt_properties.master_weights:
                    for param in optimizer._amp_stash.all_fp32_from_fp16_params:
                        param.grad = None
        else:
            optimizer.step()

        optimizer.zero_grad()

    def _prepare_dist_lamb(self, model):
        if self.config.local_rank != -1:
            flat_dist_call([param.data for param in model.parameters()], torch.distributed.broadcast, (0,))



