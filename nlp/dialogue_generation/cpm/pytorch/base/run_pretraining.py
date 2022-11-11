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

"""BERT Pretraining"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from copy import copy
import os
import time

import numpy as np
import torch
import random

import utils
from train.evaluator import Evaluator
from train.trainer import Trainer
from train.training_state import TrainingState
from config.config_manager import print_config
from dataloaders.tokenization_gpt2 import GPT2Tokenizer
from dataloaders.dataloader import load_data
from train.event import TrainingEventCompose, TrainingLogger, BaseTrainingEventInterface

logger = None

def main():
    import config
    parser = argparse.ArgumentParser("CPM")
    config.activate_config_env(parser=parser, with_config_env_name=True)

    if config.use_env and 'LOCAL_RANK' in os.environ:
        config.local_rank = int(os.environ['LOCAL_RANK'])

    interface: BaseTrainingEventInterface = config.training_event(config)
    config.training_event_instance = interface

    device, num_gpus = interface.init_distributed_environment()
    config.device = device
    config.n_gpu = num_gpus

    utils.check_config(config)

    events = [
        TrainingLogger(config, log_freq=config.log_freq)
    ]
    training_event = TrainingEventCompose(interface, events)
    training_event.launch()

    global logger
    logger = events[0].logger

    utils.barrier()
    training_event.on_init_start()
    init_start_time = logger.previous_log_time

    utils.barrier()
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)

    # get the tokenizer
    tokenizer = GPT2Tokenizer(os.path.join('dataloaders', config.tokenizer_path, 'vocab.json'), os.path.join('dataloaders', config.tokenizer_path, 'chinese_vocab.model'))
    train_dataloader, _ = load_data(config, 'train', tokenizer, 1)
    eval_dataloader, _ = load_data(config, 'valid', tokenizer, 1)
    print(f"train_dataset length:{len(train_dataloader.dataset)}")
    print(f"train length:{len(train_dataloader)}")
    print(f"eval_dataset length:{len(eval_dataloader.dataset)}")
    print(f"eval length:{len(eval_dataloader)}")

    evaluator = Evaluator(config, eval_dataloader)
    training_state = TrainingState()
    trainer = Trainer(config, training_event, evaluator, training_state, device=device)
    training_state._trainer = trainer

    utils.barrier()
    trainer.init()

    utils.barrier()
    init_evaluation_start = time.time()
    training_state.eval_avg_loss, training_state.eval_embedding_average = evaluator.evaluate(trainer)
    init_evaluation_end = time.time()
    init_evaluation_info = dict(eval_loss=training_state.eval_avg_loss,
                eval_embedding_average=training_state.eval_embedding_average,
                time=init_evaluation_end - init_evaluation_start)
    training_event.on_init_evaluate(init_evaluation_info)

    if not config.do_train:
        return config, training_state,  init_evaluation_info["time"]

    training_event.on_init_end()
    init_end_time = logger.previous_log_time
    training_state.init_time = (init_end_time - init_start_time) / 1e+3

    utils.barrier()
    epoch = -1
    training_event.on_train_begin()
    raw_train_start_time = logger.previous_log_time    
    while training_state.global_steps < config.max_steps and not training_state.end_training:
        epoch += 1
        training_state.epoch = epoch
        trainer.train_one_epoch(train_dataloader)
    training_event.on_train_end()
    raw_train_end_time = logger.previous_log_time    
    training_state.raw_train_time = (raw_train_end_time - raw_train_start_time) / 1e+3    
    
    trainer.save_checkpoint()

    return config, training_state

if __name__ == "__main__":
    now = time.time()
    config, state = main()

    if not utils.is_main_process():
        exit()

    gpu_count = config.n_gpu
    e2e_time = time.time() - now
    training_perf = (utils.global_batch_size(config) * state.global_steps) / state.raw_train_time
    if config.do_train:
        finished_info = {
            "e2e_time": e2e_time,
            "training_sequences_per_second": training_perf,
            "converged": state.converged,
            "final_loss": state.eval_avg_loss,
            "final_mlm_accuracy": state.eval_embedding_average,
            "raw_train_time": state.raw_train_time,
            "init_time": state.init_time,
        }
    else:
        finished_info = {"e2e_time": e2e_time}
    logger.log("FINISHED", finished_info, stacklevel=0)
