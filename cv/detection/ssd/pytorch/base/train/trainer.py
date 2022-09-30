import time
import os
import sys
from bisect import bisect
from typing import Union

import numpy as np
import torch
from torch.cuda.amp import GradScaler

import utils
from model import create_model
from train.evaluator import Evaluator
from train.training_state import TrainingState
from train.event import TrainingEventCompose as TrainingEvent
from box_coder import dboxes300_coco

from model.losses import OptLoss, Loss


Device = Union[torch.device, str, None]


def lr_warmup(optim, warmup_iter, iter_num, base_lr, args):
    if iter_num < warmup_iter:
        warmup_step = base_lr / (warmup_iter * (2 ** args.warmup_factor))
        new_lr = base_lr - (warmup_iter - iter_num) * warmup_step
        for param_group in optim.param_groups:
            param_group['lr'] = new_lr
        return new_lr
    else:
        return base_lr


class Trainer(object):

    def __init__(self, config, training_event: TrainingEvent,
                 evaluator: Evaluator,
                 training_state: TrainingState,
                 grad_scaler: GradScaler,
                 device: Device):
        super(Trainer, self).__init__()
        self.config = config
        self.training_event = training_event
        self.training_state = training_state
        self.grad_scaler = grad_scaler

        self.device = device
        self.optimizer = None
        self.model = None
        self.evaluator = evaluator
        self.success = torch.zeros(1).cuda()
        dboxes = dboxes300_coco()
        if self.config.dali:
            self.loss_fun = OptLoss().cuda()
        else:
            self.loss_fun = Loss(dboxes).cuda()

    def init(self):
        self.model = create_model(self.config)
        self.model = self.training_event.convert_model(self.model)
        self.optimizer = self.training_event.create_optimizer(self.model)
        self.model, self.optimizer = self.training_event.model_to_fp16(self.model, self.optimizer)
        self.model = self.training_event.model_to_ddp(self.model)
        # self.training_state.base_lr = self.optimizer.param_groups[0]['lr']
        self.training_state.base_lr = self.optimizer.defaults['lr']
        if utils.is_main_process():
            print("==="*20)
            print("config lr: {}, optimizer lr: {}".format(self.config.learning_rate, self.training_state.base_lr))
            print("==="*20)

        self._init_model()
        self.model.train()
        self._verify_model()

    def _init_model(self):
        if self.config.checkpoint:
            checkpoint = torch.load(self.config.checkpoint, map_location="cpu")
            self.training_event.load_checkpoint(checkpoint)
            self.training_state.iter_num = self.config.iteration

    def _verify_model(self):
        input_c = 4 if self.config.pad_input else 3
        if self.config.nhwc:
            example_shape = [self.config.train_batch_size, 300, 300, input_c]
        else:
            example_shape = [self.config.train_batch_size, input_c, 300, 300]
        example_input = torch.randn(*example_shape).cuda()
        if self.config.fp16:
            example_input = example_input.half()
        if self.config.jit:
            # DDP has some Python-side control flow.  If we JIT the entire DDP-wrapped module,
            # the resulting ScriptModule will elide this control flow, resulting in allreduce
            # hooks not being called.  If we're running distributed, we need to extract and JIT
            # the wrapped .module.
            # Replacing a DDP-ed ssd300 with a script_module might also cause the AccumulateGrad hooks
            # to go out of scope, and therefore silently disappear.
            module_to_jit = self.model.module if self.config.distributed else self.model
            if self.config.distributed:
                self.model.module = torch.jit.trace(module_to_jit, example_input, check_trace=False)
            else:
                self.model = torch.jit.trace(module_to_jit, example_input, check_trace=False)
        ploc, plabel = self.model(example_input)
        loss = ploc[0, 0, 0] + plabel[0, 0, 0]
        dloss = torch.randn_like(loss)
        # Cause cudnnFind for dgrad, wgrad to run
        loss.backward(dloss)
        for p in self.model.parameters():
            p.grad = None

    def train_one_epoch(self, train_dataloader):
        if self.training_state.epoch in self.config.lr_decay_epochs:
            self.training_state.base_lr *= self.config.lr_decay_factor
            print(self.config.local_rank, "base_lr decay step #" + str(bisect(self.config.lr_decay_epochs, self.training_state.epoch)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.training_state.base_lr
        if self.training_state.epoch <= self.config.epoch:
            print("Start continue training from epoch: {}, iter: {}, skip epoch {}".format(self.config.epoch, self.config.iteration, self.training_state.epoch))
            return

        self.training_event.on_epoch_begin(self.training_state.epoch)
        step_start_time = time.time()
        for batch in train_dataloader:
            # print([len(x) for x in batch])
            img, bbox, label = batch
            if not self.config.dali:
                img = img.cuda()
                bbox = bbox.cuda()
                label = label.cuda()

            self.training_state.lr = lr_warmup(self.optimizer, self.config.warmup, self.training_state.iter_num, self.training_state.base_lr, self.config)
            if (img is None) or (bbox is None) or (label is None):
                print("No labels in batch")
                continue
            self.training_event.on_step_begin(self.training_state.iter_num)
            self.train_one_step(img, bbox, label)

            other_state = dict()
            if self.training_state.iter_num % self.config.gradient_accumulation_steps == 0:
                step_end_time = time.time()
                step_total_time = step_end_time - step_start_time
                fps = (utils.global_batch_size(self.config) * self.config.gradient_accumulation_steps) / step_total_time
                other_state["avg_samples/s"] = fps
                step_start_time = step_end_time

            step_info = self.training_state.to_dict(**other_state)
            self.training_event.on_step_end(self.training_state.iter_num, result=step_info)
            self.training_state.iter_num += 1

        if self.config.dali:
            train_dataloader.reset()
        if self.training_state.epoch in self.config.evaluation:
            if self.config.distributed:
                world_size = float(utils.get_world_size())
                for bn_name, bn_buf in self.model.module.named_buffers(recurse=True):
                    if ('running_mean' in bn_name) or ('running_var' in bn_name):
                        torch.distributed.all_reduce(bn_buf, op=torch.distributed.ReduceOp.SUM)
                        bn_buf /= world_size

            eval_start = time.time()
            self.training_state.eval_ap = self.evaluator.evaluate(self)
            eval_end = time.time()
            eval_result = dict(epoch=self.training_state.epoch,
                               eval_ap=self.training_state.eval_ap,
                               time=eval_end - eval_start)
            self.training_event.on_evaluate(eval_result)
            if utils.is_main_process():
                if self.config.save_checkpoint:
                    print("saving model...")
                    if not os.path.isdir(self.config.output):
                        os.mkdir(self.config.output)
                    self.training_event.save_checkpoint(self.config.output, self.training_state)
                self.detect_training_status(self.training_state)
                if self.training_state.converged:
                    self.success = torch.ones(1).cuda()
            if self.config.distributed:
                torch.distributed.broadcast(self.success, 0)
            if self.success[0]:
                print("Process {} train success!".format(self.config.local_rank))
                self.training_state.end_training = True
        self.training_event.on_epoch_end(self.training_state.epoch)

    def train_one_step(self, img, bbox, label):
        self.training_state.loss, _, _ = self.forward(img, bbox, label)
        if self.training_state.epoch == self.config.epoch + 1 and self.training_state.iter_num == self.config.iteration:
            self.training_state.avg_loss = self.training_state.loss.item()
        else:
            if np.isfinite(self.training_state.loss.item()):
                self.training_state.avg_loss = 0.999 * self.training_state.avg_loss + 0.001 * self.training_state.loss.item()
            else:
                print("model exploded (corrupted by Inf or Nan)")
                sys.exit()
        self.training_event.on_backward(self.training_state.iter_num, self.training_state.loss, self.optimizer, self.grad_scaler)

    def forward(self, img, bbox, label, training=True):
        # origin input shape is: (bs, 3, 300, 300)
        # using dali and nhwc input shape is: (bs, 300, 300, 4)
        ploc, plabel = self.model(img)
        ploc, plabel = ploc.float(), plabel.float()
        if training:
            N = img.shape[0]
            bbox.requires_grad = False
            label.requires_grad = False
            # reshape (N*8732X4 -> Nx8732x4) and transpose (Nx8732x4 -> Nx4x8732)
            bbox = bbox.view(N, -1, 4).transpose(1, 2).contiguous()
            # reshape (N*8732 -> Nx8732) and cast to Long
            label = label.view(N, -1).long()

            # torch.save({
            #     "ploc": ploc,
            #     "plabel": plabel,
            #     "gloc": bbox,
            #     "glabel": label,
            # }, "loss.pth_{}".format(self.config.local_rank))
            # exit()
            loss = self.loss_fun(ploc, plabel, bbox, label)
            return loss, None, None
        else:
            return None, ploc, plabel

    def inference(self, img):
        return self.forward(img, None, None, False)

    def detect_training_status(self, training_state: TrainingState):
        if training_state.eval_ap >= self.config.threshold:
            training_state.converged_success()
        if training_state.converged or training_state.epoch >= self.config.epochs:
            training_state.end_training = True
        return training_state.end_training