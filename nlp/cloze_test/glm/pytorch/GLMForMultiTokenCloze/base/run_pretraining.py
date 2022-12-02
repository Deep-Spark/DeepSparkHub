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
import time
import argparse
import os
import numpy as np
import torch
import random

from dataloaders import (WorkerInitializer,
                         build_train_dataloader,
                         build_eval_dataloaders)
import utils
from train import Trainer, Evaluator
from train.training_state import TrainingState
from train.event import TrainingEventImpl, TrainingLogger

logger = None


def main():
    import config
    parser = argparse.ArgumentParser("Glm")
    config.activate_config_env(parser=parser, with_config_env_name=True)

    if config.use_env and 'LOCAL_RANK' in os.environ:
        config.local_rank = int(os.environ['LOCAL_RANK'])

    device, num_gpus = utils.init_dist_training_env(config)
    config.device = device
    config.n_gpu = num_gpus

    utils.check_config(config)

    interface = config.training_event(config)
    events = [
        TrainingLogger(config, log_freq=config.log_freq)
    ]
    training_event = TrainingEventImpl(interface, events)
    training_event.launch()

    global logger
    logger = events[0].logger

    utils.barrier()
    training_event.on_init_start()
    init_start_time = logger.previous_log_time

    worker_seeds, shuffling_seeds = utils.setup_seeds(
        config.seed, config.num_epochs_to_generate_seeds_for, device)

    if torch.distributed.is_initialized():
        worker_seed = worker_seeds[torch.distributed.get_rank()]
    else:
        worker_seed = worker_seeds[0]

    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    worker_init = WorkerInitializer.default(worker_seed)

    evaluator = Evaluator(config, None)
    training_state = TrainingState()
    trainer = Trainer(training_event=training_event, evaluator=evaluator,
                      training_state=training_state, config=config)
    training_state._trainer = trainer

    utils.barrier()
    trainer.init()

    eval_dataloader = build_eval_dataloaders(config)

    utils.barrier()
    init_evaluation_start = time.time()
    evaluator.dataloader = eval_dataloader
    score = trainer.evaluator.evaluate(trainer)
    training_state.eval_accuracy = score
    init_evaluation_end = time.time()
    init_evaluation_info = dict(
        eval_accuracy=score,
        time=init_evaluation_end - init_evaluation_start
    )
    training_event.on_init_evaluate(init_evaluation_info)


    train_dataloader = build_train_dataloader(config, worker_init)

    if not config.do_train:
        return config, training_state

    training_event.on_init_end()
    init_end_time = logger.previous_log_time
    training_state.init_time = (init_end_time - init_start_time) / 1e+3

    utils.barrier()

    epoch = -1
    training_event.on_train_begin()
    raw_train_start_time = logger.previous_log_time
  
    while training_state.num_trained_samples < config.max_samples_termination and not training_state.end_training:
        epoch += 1
        training_state.epoch = epoch
        train_dataloader.sampler.set_epoch(epoch)
        trainer.train_one_epoch(train_dataloader)
    
    training_event.on_train_end()
    raw_train_end_time = logger.previous_log_time
    training_state.raw_train_time = (raw_train_end_time - raw_train_start_time) / 1e+3

    return config, training_state


if __name__ == "__main__":

    now = time.time()
    config, state = main()
    
    if not utils.is_main_process():
        exit()
    
    gpu_count = config.n_gpu
    e2e_time = time.time() - now
    if config.do_train:
        training_perf = (utils.global_batch_size(config) * state.global_steps) / state.raw_train_time
        finished_info = {
            "e2e_time": e2e_time,
            "training_sequences_per_second": training_perf,
            "converged": state.converged,
            "final_accuracy": state.eval_accuracy,
            "raw_train_time": state.raw_train_time,
            "init_time": state.init_time,
        }
    else:
        finished_info = {"e2e_time": e2e_time}
    logger.log("FINISHED", finished_info, stacklevel=0)
