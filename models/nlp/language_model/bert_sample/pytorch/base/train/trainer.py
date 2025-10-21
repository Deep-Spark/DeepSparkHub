# Copyright (c) 2022 Iluvatar CoreX. All rights reserved.
# Copyright Declaration: This software, including all of its code and documentation,
# except for the third-party software it contains, is a copyrighted work of Shanghai Iluvatar CoreX
# Semiconductor Co., Ltd. and its affiliates ("Iluvatar CoreX") in accordance with the PRC Copyright
# Law and relevant international treaties, and all rights contained therein are enjoyed by Iluvatar
# CoreX. No user of this software shall have any right, ownership or interest in this software and
# any use of this software shall be in compliance with the terms and conditions of the End User
# License Agreement.

import numpy as np
import math
import time
import os
import sys

import torch
from torch.cuda.amp import GradScaler
from torch.types import Device

import config
import utils
from dataloaders.dataset import exchange_padding_fast
from model import create_model
from schedulers import create_scheduler
from train.evaluator import Evaluator
from train.training_state import TrainingState
from train.event import TrainingEventCompose as TrainingEvent
from utils.checkpoint import remap_segmented_model_parameters


class Trainer():

    def __init__(self, training_event: TrainingEvent,
                 evaluator: Evaluator,
                 training_state: TrainingState,
                 grad_scaler: GradScaler,
                 device: Device):
        super(Trainer, self).__init__()
        self.training_event = training_event
        self.training_state = training_state
        self.grad_scaler = grad_scaler

        self.device = device
        self.optimizer = None
        self.bert_config = None
        self.model = None
        self.evaluator = evaluator
        self.lr_scheduler = None
        self.init_epoch = 0
        self.init_dataloader_idx = 0

    def init(self):
        self.bert_config, self.model = create_model(config)
        self.model = self._init_model(self.model, self.device)
        self.model = self.training_event.convert_model(self.model)
        self.optimizer = self.training_event.create_optimizer(self.model)
        self.model, self.optimizer = self.training_event.model_to_fp16(self.model, self.optimizer)
        self.model = self.training_event.model_to_ddp(self.model)
        self.lr_scheduler = create_scheduler(self.optimizer)
        self.load()

    def _init_model(self, model, device):
        checkpoint = torch.load(config.init_checkpoint, map_location="cpu")
        if self._is_resume_checkpoint(checkpoint):
            if "global_steps" in checkpoint['state'] and checkpoint['state']['global_steps'] > 0:
                config.learning_rate = checkpoint['state']['learning_rate']
        else:
            if "model" in checkpoint:
                checkpoint = checkpoint["model"]
            checkpoint_remapped = remap_segmented_model_parameters(checkpoint)
            model.load_state_dict(checkpoint_remapped, strict=True)
        model = model.to(device)
        return model

    def _is_resume_checkpoint(self, checkpoint):
        return "optimizer" in checkpoint
    
    def load(self):
        checkpoint = torch.load(config.init_checkpoint, map_location="cpu")

        if not self._is_resume_checkpoint(checkpoint):
            return

        model_ckpt = checkpoint['model']
        if config.distributed_lamb:
            self.model.load_state_dict(model_ckpt,strict=True)
        else:
            self.model.module.load_state_dict(model_ckpt,strict=True)

        if 'global_steps' in  checkpoint['state'] and checkpoint['state']['global_steps'] > 0:
            # restore optimizer
            optimizer_ckpt = checkpoint['optimizer']
            self.optimizer.load_state_dict(optimizer_ckpt)

            # restore epoch, dataloader_idx
            self.init_epoch = checkpoint['state']['epoch']
            self.init_dataloader_idx = checkpoint['state']['iter_dataloader_idx']

            self.training_state.global_steps = checkpoint['state']['global_steps']
        

    def save(self):
        model_dict = self.model.state_dict()
        optimizer_dict = self.optimizer.state_dict()
        state_dict = self.training_state.to_dict()

        save_dict = {'model':model_dict,'optimizer':optimizer_dict,'state': state_dict}

        save_file = os.path.join(config.output_dir,f'model.ckpt-{self.training_state.global_steps}.pt')

        utils.main_proc_print(f"save for steps:{self.training_state.global_steps}")

        if utils.get_rank() == 0 or utils.get_rank() == -1:
            if not os.path.isdir(config.output_dir):
                os.makedirs(config.output_dir)
            torch.save(save_dict,save_file)

    def train_one_epoch(self, dataloader):
        state = self.training_state
        training_event = self.training_event

        # restore epoch
        if state.epoch < self.init_epoch:
            return

        training_event.on_epoch_begin(state.epoch)

        step_start_time = time.time()
        for dataloader_idx, batch_idx, batch in dataloader.iter_batchs():

            # restore dataloader
            if state.epoch == self.init_epoch and dataloader_idx <= self.init_dataloader_idx:
                continue

            state.num_trained_samples = state.global_steps * utils.global_batch_size(config)

            state.global_steps += 1
            state.iter_dataloader_idx = dataloader_idx
            self.training_event.on_step_begin(state.global_steps)
            self.train_one_step(batch_idx, batch)

            other_state = dict()
            if state.global_steps % config.gradient_accumulation_steps == 0:
                step_end_time = time.time()
                step_total_time = step_end_time - step_start_time
                step_start_time = step_end_time
                sequences_per_second = (utils.global_batch_size(config) * config.gradient_accumulation_steps) / step_total_time
                other_state["seq/s"] = sequences_per_second

            eval_result = None
            # if self.can_do_eval(state):
            if state.global_steps > 0 and  state.global_steps % config.eval_steps == 0:
                eval_start = time.time()
                state.eval_loss, state.eval_mlm_accuracy = self.evaluator.evaluate(self)
                eval_end = time.time()
                eval_result = dict(global_steps=state.global_steps,
                                   eval_loss=state.eval_loss,
                                   eval_mlm_accuracy=state.eval_mlm_accuracy,
                                   time=eval_end - eval_start)
                if config.save_checkpoint:
                    self.save()

            end_training = self.detect_training_status(state)

            step_info = state.to_dict(**other_state)

            self.training_event.on_step_end(state.global_steps, result=step_info)

            if eval_result is not None:
                self.training_event.on_evaluate(eval_result)

            if end_training:
                break

        training_event.on_epoch_end(state.epoch)

    def train_one_step(self, batch_idx, batch):
        if config.exchange_padding == True:
            batch = [t.to(self.device, non_blocking=True, dtype=torch.int16) for t in batch]
            batch = exchange_padding_fast(self.device, config.train_batch_size, *batch)
        else:
            batch = [t.to(self.device, non_blocking=True) for t in batch]

        state = self.training_state

        self.model.train()
        state.loss, state.mlm_acc, _ = self.forward(batch)
        if not np.isfinite(state.loss.item()):
            print("Loss is {}, stopping training".format(state.loss.item()))
            sys.exit(1)
        self.training_event.on_backward(state.global_steps, state.loss, self.optimizer, self.grad_scaler)
        self.lr_scheduler.step()

    def detect_training_status(self, state: TrainingState):
        if state.eval_mlm_accuracy >= config.target_mlm_accuracy:
            state.converged_success()

        if state.global_steps > config.max_steps or state.num_trained_samples > config.max_samples_termination:
            state.end_training = True

        return state.end_training

    def can_do_eval(self, state: TrainingState):
        do_eval = all([
            config.eval_dir is not None,
            state.num_trained_samples >= config.eval_iter_start_samples,
            state.global_steps % math.ceil(config.eval_interval_samples / utils.global_batch_size(config)) == 0,
            config.eval_interval_samples > 0,
            state.global_steps > 1,
        ])

        return do_eval or state.global_steps >= config.max_steps

    def forward(self, batch):
        input_ids, segment_ids, input_mask, masked_lm_labels, next_sentence_labels = batch
        loss, mlm_acc, num_valid = self.model(input_ids, segment_ids, input_mask,
                                              masked_lm_labels, next_sentence_labels)
        return loss, mlm_acc, num_valid

    def inference(self, batch):
        self.model.eval()
        return self.forward(batch)
