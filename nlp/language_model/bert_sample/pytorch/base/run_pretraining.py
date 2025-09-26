# Copyright (c) 2022 Iluvatar CoreX. All rights reserved.
# Copyright Declaration: This software, including all of its code and documentation,
# except for the third-party software it contains, is a copyrighted work of Shanghai Iluvatar CoreX
# Semiconductor Co., Ltd. and its affiliates ("Iluvatar CoreX") in accordance with the PRC Copyright
# Law and relevant international treaties, and all rights contained therein are enjoyed by Iluvatar
# CoreX. No user of this software shall have any right, ownership or interest in this software and
# any use of this software shall be in compliance with the terms and conditions of the End User
# License Agreement.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from copy import copy
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
from torch.cuda.amp import GradScaler

import utils
from dataloaders import WorkerInitializer
from dataloaders.dataloader import PretrainingDataloaders
from train.evaluator import Evaluator
from train.trainer import Trainer
from train.training_state import TrainingState
from train.event import TrainingEventCompose, TrainingLogger


logger = None


def main():
    import config
    parser = argparse.ArgumentParser("Bert")
    config.activate_config_env(parser=parser, with_config_env_name=True)

    if config.use_env and 'LOCAL_RANK' in os.environ:
        config.local_rank = int(os.environ['LOCAL_RANK'])

    device, num_gpus = utils.init_dist_training_env(config)
    config.device = device
    config.n_gpu = num_gpus

    utils.check_config(config)

    try:
        from dltest import show_training_arguments
        show_training_arguments(config)
    except:
        pass

    interface = config.training_event(config)
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
    worker_seeds, shuffling_seeds = utils.setup_seeds(config.seed, config.num_epochs_to_generate_seeds_for, device)
    if torch.distributed.is_initialized():
        worker_seed = worker_seeds[torch.distributed.get_rank()]
    else:
        worker_seed = worker_seeds[0]

    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    worker_init = WorkerInitializer.default(worker_seed)

    pool = ProcessPoolExecutor(1)
    evaluator = Evaluator(
        config.eval_dir,
        proc_pool=pool,
        global_batch_size=utils.global_batch_size(config),
        max_steps=config.max_steps,
        worker_init=worker_init,
        use_cache=config.cache_eval_data
    )
    grad_scaler = GradScaler(init_scale=float(os.getenv("INIT_LOSS_SCALE", 2 ** 20)), growth_interval=2000)
    training_state = TrainingState()
    trainer = Trainer(training_event, evaluator, training_state, grad_scaler, device=device)
    training_state._trainer = trainer

    utils.barrier()
    trainer.init()

    utils.barrier()
    init_evaluation_start = time.time()
    eval_loss, eval_mlm_acc = evaluator.evaluate(trainer)
    training_state.eval_loss = eval_loss
    training_state.eval_mlm_accuracy = eval_mlm_acc
    init_evaluation_end = time.time()
    init_evaluation_info = dict(
        eval_loss = eval_loss,
        eval_mlm_accuracy = eval_mlm_acc,
        time = init_evaluation_end - init_evaluation_start
    )
    training_event.on_init_evaluate(init_evaluation_info)

    if not config.do_train:
        return config, training_state, init_evaluation_info["time"]

    dataloader = PretrainingDataloaders(
        config.train_dir,
        max_predictions_per_seq=config.max_predictions_per_seq,
        batch_size=config.train_batch_size,
        seed=shuffling_seeds, num_files_per_iter=1,
        worker_init=worker_init, pool=pool,
    )

    training_event.on_init_end()
    init_end_time = logger.previous_log_time
    training_state.init_time = (init_end_time - init_start_time) / 1e+3

    utils.barrier()

    epoch = -1
    training_event.on_train_begin()
    raw_train_start_time = logger.previous_log_time
    if config.save_checkpoint:
        trainer.save()
    while training_state.global_steps < config.max_steps and not training_state.end_training:
        epoch += 1
        training_state.epoch = epoch
        dataloader.set_epoch(epoch)
        trainer.train_one_epoch(dataloader)
    if config.save_checkpoint:
        trainer.save()
    training_event.on_train_end()
    raw_train_end_time = logger.previous_log_time
    training_state.raw_train_time = (raw_train_end_time - raw_train_start_time) / 1e+3
    return config, training_state


if __name__ == "__main__":
    now = time.time()
    config, state = main()

    if not utils.is_main_process():
        exit(1)

    gpu_count = config.n_gpu
    e2e_time = time.time() - now
    training_perf = (utils.global_batch_size(config) * state.global_steps) / (state.raw_train_time + 1e-7)
    if config.do_train:
        finished_info = {
            "e2e_time": e2e_time,
            "training_sequences_per_second": training_perf,
            "converged": state.converged,
            "final_loss": state.eval_loss,
            "final_mlm_accuracy": state.eval_mlm_accuracy,
            "raw_train_time": state.raw_train_time,
            "init_time": state.init_time,
        }
    else:
        finished_info = {"e2e_time": e2e_time}
    logger.log("FINISHED", finished_info, stacklevel=0)
    if state.converged:
        exit(0)
    else:
        exit(1)
