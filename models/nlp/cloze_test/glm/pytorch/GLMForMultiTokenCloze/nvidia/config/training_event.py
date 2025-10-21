# Copyright (c) 2022, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# All Rights Reserved.
#
#    Licensed under the Apache License, Version 2.0 (the "License"); you may
#    not use this file except in compliance with the License. You may obtain
#    a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
#    WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
#    License for the specific language governing permissions and limitations
#    under the License.
import torch

from train.event.base import BaseTrainingEventInterface
from utils import main_proc_print
from utils import PyTorchDistributedDataParallel as TorchDDP
from model.models.modeling import FP16_Module
from create_optimizer import create_optimizer
from optimizers.loss_scaler import DynamicLossScaler
from converter import convert_model

clip_grad_norm = torch.nn.utils.clip_grad_norm_


class DefaultTrainingEvent(BaseTrainingEventInterface):
    def __init__(self, config):
        super(DefaultTrainingEvent, self).__init__(config)
        self.config = config

    def model_to_fp16(self, model):
        # To prevent OOM for model sizes that cannot fit in GPU memory in full precision
        if self.config.fp16:
            main_proc_print(" > use fp16...")
            model.half()

        # GPU allocation.
        model.cuda(torch.cuda.current_device())

        # Fp16 conversion.
        if self.config.fp16:
            model = FP16_Module(model)
        return model

    def model_to_ddp(self, model):
        i = torch.cuda.current_device()
        model = TorchDDP(model, device_ids=[i], output_device=i)
        self.model = model
        return model

    def create_optimizer(self, model):
        return create_optimizer(model, self.config)

    def on_backward(self, step, lm_loss, reduced_loss, optimizer, lr_scheduler):
        args = self.config

        if not DynamicLossScaler._has_inf_or_nan(reduced_loss):
            backward_step(optimizer, self.model, lm_loss, args)
            if step % args.gradient_accumulation_steps == 0:
                optimizer.step()
                if not (args.fp16 and optimizer.overflow):
                    lr_scheduler.step()
                optimizer.zero_grad()

        else:
            main_proc_print("Found NaN loss, skip backward")
        return reduced_loss
    
    def convert_model(self, model):
        return convert_model(model,self.config)


def backward_step(optimizer, model, lm_loss, args):
    """Backward step."""

    # Total loss.
    loss = lm_loss

    if args.fp16:
        optimizer.backward(loss, update_master_grads=False)
    else:
        loss.backward()

    if args.fp16:
        optimizer.update_master_grads()

    # Clipping gradients helps prevent the exploding gradient.
    if args.clip_grad > 0:
        if not args.fp16:
            clip_grad_norm(model.parameters(), args.clip_grad)
        else:
            optimizer.clip_master_grads(args.clip_grad)

    return lm_loss
