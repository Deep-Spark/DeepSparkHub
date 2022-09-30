"""SSD training"""

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
from dataloaders.dataloader import create_train_dataloader
from train.evaluator import Evaluator
from train.trainer import Trainer
from train.training_state import TrainingState
from train.event import TrainingEventCompose, TrainingLogger

logger = None


def main():
    import config
    parser = argparse.ArgumentParser("SSD")
    config.activate_config_env(parser=parser, with_config_env_name=True)

    if config.use_env and 'LOCAL_RANK' in os.environ:
        config.local_rank = int(os.environ['LOCAL_RANK'])

    device, num_gpus = utils.init_dist_training_env(config)
    config.device = device
    config.n_gpu = num_gpus

    utils.check_config(config)

    torch.backends.cudnn.benchmark = True
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
    utils.setup_seeds(config)

    evaluator = Evaluator(config)
    grad_scaler = GradScaler(init_scale=float(os.getenv("INIT_LOSS_SCALE", 2 ** 20)), growth_interval=2000)
    training_state = TrainingState()
    trainer = Trainer(config, training_event, evaluator, training_state, grad_scaler, device=device)
    training_state._trainer = trainer

    train_dataloader, epoch_size, train_sampler = create_train_dataloader(config)
    utils.barrier()
    trainer.init()

    utils.barrier()
    init_evaluation_start = time.time()
    eval_ap = evaluator.evaluate(trainer)
    training_state.eval_ap = eval_ap
    init_evaluation_end = time.time()
    init_evaluation_info = dict(
        eval_ap=eval_ap,
        time=init_evaluation_end - init_evaluation_start
    )
    training_event.on_init_evaluate(init_evaluation_info)

    if not config.do_train:
        return config, training_state, init_evaluation_info["time"]

    training_event.on_init_end()
    init_end_time = logger.previous_log_time
    training_state.init_time = (init_end_time - init_start_time) / 1e+3

    utils.barrier()

    training_event.on_train_begin()
    raw_train_start_time = logger.previous_log_time
    while training_state.epoch < config.epochs and not training_state.end_training:
        training_state.epoch += 1
        trainer.train_one_epoch(train_dataloader)
        if config.distributed and not config.dali:
            train_sampler.set_epoch(training_state.epoch)
    training_event.on_train_end()
    raw_train_end_time = logger.previous_log_time
    training_state.raw_train_time = (raw_train_end_time - raw_train_start_time) / 1e+3
    return config, training_state, epoch_size


if __name__ == "__main__":
    now = time.time()
    config, training_state, epoch_size = main()

    if not utils.is_main_process():
        print("process {} exit!".format(config.local_rank))
        exit()

    gpu_count = config.n_gpu
    e2e_time = time.time() - now
    training_perf = (epoch_size * training_state.epoch) / training_state.raw_train_time
    if config.do_train:
        finished_info = {
            "e2e_time": e2e_time,
            "training_samples_per_second": training_perf,
            "converged": training_state.converged,
            "final_ap": training_state.eval_ap,
            "raw_train_time": training_state.raw_train_time,
            "init_time": training_state.init_time,
        }
    else:
        finished_info = {"e2e_time": e2e_time}
    logger.log("FINISHED", finished_info, stacklevel=0)
