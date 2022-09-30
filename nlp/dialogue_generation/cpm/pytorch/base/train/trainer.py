import math
import time
import os

import torch
from torch.types import Device

import utils
from model import create_model
from schedulers import create_scheduler
from train.evaluator import Evaluator
from train.training_state import TrainingState
from train.event import TrainingEventCompose as TrainingEvent
from train.metrics import average_corpus_level
from model.losses.cross_entropy import cross_entropy
from model.fp16 import FP16_Module

class Trainer():

    def __init__(self, config, training_event: TrainingEvent,
                 evaluator: Evaluator,
                 training_state: TrainingState,
                 device: Device):
        super(Trainer, self).__init__()
        self.config = config
        self.training_event = training_event
        self.training_state = training_state
        self.device = device
        self.optimizer = None
        self.model = None
        self.evaluator = evaluator
        self.lr_scheduler = None

    def init(self):
        self.model_config, self.model = create_model(self.config)
        self.model = self._init_model(self.model, self.device)
        self.model = self.training_event.convert_model(self.model)
        self.model = self.model.to(self.config.device)

        self.optimizer = self.training_event.create_optimizer(self.model)
        self.model, self.optimizer = self.training_event.model_to_fp16(self.model, self.optimizer)
        self.model = self.training_event.model_to_ddp(self.model)

        self.lr_scheduler = create_scheduler(self.optimizer, self.config)

    def _init_model(self, model, device):
        checkpoint = torch.load(self.config.init_checkpoint, map_location="cpu")
        if "model" in checkpoint:
            checkpoint = checkpoint["model"]
        
        model.load_state_dict(checkpoint, strict=True)

        return model

    def train_one_epoch(self, dataloader):
        state = self.training_state
        training_event = self.training_event

        training_event.on_epoch_begin(state.epoch)

        step_start_time = time.time()
        for  _, data in enumerate(dataloader):
            batch, no_model_batch = data[0], data[1]

            state.global_steps += 1
            state.num_trained_samples = state.global_steps * utils.global_batch_size(self.config)

            self.training_event.on_step_begin(state.global_steps)            
            self.train_one_step(batch, no_model_batch)

            other_state = dict()
            if state.global_steps % self.config.gradient_accumulation_steps == 0:
                step_end_time = time.time()
                step_total_time = step_end_time - step_start_time
                step_start_time = step_end_time
                sequences_per_second = (utils.global_batch_size(self.config) * self.config.gradient_accumulation_steps) / step_total_time
                other_state["seq/s"] = sequences_per_second

            eval_result = None
            if self.can_do_eval(state):
                eval_start = time.time()                
                state.eval_avg_loss, state.eval_embedding_average = self.evaluator.evaluate(self)
                eval_end = time.time()                
                eval_result = dict(global_steps=state.global_steps,
                         eval_loss=state.eval_avg_loss,
                         eval_embedding_average=state.eval_embedding_average,
                        time=eval_end - eval_start)
 
            end_training = self.detect_training_status(state)

            step_info = state.to_dict(**other_state)
            self.training_event.on_step_end(state.global_steps, result=step_info)

            if eval_result is not None:
                self.training_event.on_evaluate(eval_result)

            if end_training:
                break

        training_event.on_epoch_end(state.epoch)

    def train_one_step(self, batch, no_model_batch):
        for k in batch:
            batch[k] = batch[k].to(self.device)
        for k in no_model_batch:
            no_model_batch[k] = no_model_batch[k].to(self.device)

        state = self.training_state
        self.model.train()

        output = self.model(**batch)
        labels = no_model_batch["labels"]

        #losses 的形状：[b,s]
        losses = cross_entropy(output.contiguous().float(), labels)
        loss_mask = no_model_batch["loss_mask"].view(-1)
        #loss 为标量
        state.loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

        preds = torch.argmax(output, -1)
        if isinstance(self.model.module, FP16_Module):
            embeddings = self.model.module.module.word_embeddings.weight
        else:
            embeddings = self.model.module.word_embeddings.weight

        #embedding_average 形状是[batch_size]
        embedding_average = average_corpus_level(preds.cpu().detach(), labels.cpu().detach(), embeddings.cpu().detach(), no_model_batch["loss_mask"].cpu().detach())
        state.embedding_average = float(embedding_average.mean)

        self.training_event.on_backward(state.global_steps, state.loss, self.optimizer)
        self.lr_scheduler.step()


    def detect_training_status(self, state: TrainingState):
        if state.eval_embedding_average >= self.config.target_embedding_average:
            state.converged_success()

        if state.global_steps >= self.config.max_steps or state.num_trained_samples >= self.config.max_samples_termination:
            state.end_training = True

        return state.end_training

    def can_do_eval(self, state: TrainingState):
        do_eval = all([
            self.config.data_dir is not None,
            state.num_trained_samples >= self.config.eval_iter_start_samples,
            self.config.eval_interval_samples > 0,
            state.global_steps > 1,
            state.global_steps % math.ceil(self.config.eval_interval_samples / utils.global_batch_size(self.config)) == 0,
        ])

        return do_eval or state.global_steps >= self.config.max_steps
