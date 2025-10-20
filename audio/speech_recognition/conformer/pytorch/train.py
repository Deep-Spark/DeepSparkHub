# Copyright (c) 2022 Iluvatar CoreX. All rights reserved.
# Copyright Declaration: This software, including all of its code and documentation,
# except for the third-party software it contains, is a copyrighted work of Shanghai Iluvatar CoreX
# Semiconductor Co., Ltd. and its affiliates ("Iluvatar CoreX") in accordance with the PRC Copyright
# Law and relevant international treaties, and all rights contained therein are enjoyed by Iluvatar
# CoreX. No user of this software shall have any right, ownership or interest in this software and
# any use of this software shall be in compliance with the terms and conditions of the End User
# License Agreement.

from collections import OrderedDict
import numpy as np
import os
import sys
import time
from tqdm import trange

import torch
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
from torch.nn.parallel import DistributedDataParallel as DDP

from openspeech.tokenizers import TOKENIZER_REGISTRY
from openspeech.datasets import DATA_MODULE_REGISTRY
from openspeech.models import MODEL_REGISTRY

from dataloader import create_dataloader, WorkerInitializer
from utils import ConfigDict, dist, manual_seed, TrainingLogger, TrainingState
import math

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(
        description="This is an example for using TrainingLogger.")
    parser.add_argument("--config_file", type=str, default="configs/conformer_lstm.json")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--max_steps", type=int, default=10000,
                        help="Total number of training steps to perform.")
    parser.add_argument("--max_epochs", type=int, default=None,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Total batch size for training.")
    parser.add_argument("--quality_metric", type=str, default="wer")
    parser.add_argument("--quality_target", type=float, default=0.99)
    parser.add_argument("--quality_judgement", type=str, default='<=')
    parser.add_argument("--num_train_samples", type=int, default=None,
                        help="Number of train samples to run train on.")
    parser.add_argument("--num_eval_samples", type=int, default=None,
                        help="Number of eval samples to run eval on.")
    parser.add_argument("--eval_batch_size", type=int, default=None,
                        help="Total batch size for evaluation.")
    parser.add_argument("--eval_freq", type=int, default=1000,
                        help="Evaluate every eval_freq steps during training.")
    parser.add_argument("--log_freq", type=int, default=10,
                        help="Frequency of logging training state.")
    parser.add_argument('--seed', type=int, default=1234,
                        help="Random seed for initialization")
    parser.add_argument("--amp", action="store_true",
                        help="Use mixed accuracy training.")
    parser.add_argument("--ddp", action="store_true",
                        help="Use distributed training.")
    parser.add_argument("--local_rank", "--local-rank", type=int, default=-1,
                        help="Local rank for distributed training on gpus.")
    parser.add_argument("--ddp_type", default='native', type=str)
    parser.add_argument("--dist_backend", type=str, default="nccl",
                        help="Communication backend for distributed training on gpus.")

    args = parser.parse_args()
    return args


def h2d_tensors(tensors, device):
    return [t.to(device) for t in tensors]


class StrFormatter():
    fmt_dict = dict(
        stage=' [{0: <12}]',
        progress=' [{0: <19}]',
        metrics=' [{0: <15}]',
        perf=' [{0: <12}]',
        default=' {0: <10}'
    )

    def __call__(self, key, str_msg):
        if key not in self.fmt_dict:
            key = 'default'
        return self.fmt_dict[key].format(str_msg)


def main(args):
    import json

    with open(args.config_file) as f:
        configs = json.load(f)

    configs = ConfigDict(configs)
    configs.dataset.dataset_path = os.path.join(
        args.data_dir, configs.dataset.dataset_path)
    configs.dataset.train_manifest_file = os.path.join(
        args.data_dir, configs.dataset.train_manifest_file)
    configs.dataset.eval_manifest_file = os.path.join(
        args.data_dir, configs.dataset.eval_manifest_file)
    configs.tokenizer.vocab_path = os.path.join(
        args.data_dir, configs.tokenizer.vocab_path)

    logger = TrainingLogger(
        flush_freq=1,
        json_flush_freq=10,
        filepath='training_log.json',
        str_formatter=StrFormatter()
    )

    args.device, args.num_gpus = dist.init_dist_training_env(args)

    worker_seeds, shuffling_seeds = dist.setup_seeds(
        args.seed, 1, args.device)
    if args.ddp:
        worker_seed = worker_seeds[torch.distributed.get_rank()]
    else:
        worker_seed = worker_seeds[0]
    manual_seed(worker_seed)
    worker_init = WorkerInitializer.default(worker_seed)

    tokenizer = TOKENIZER_REGISTRY[configs.tokenizer.unit](configs)

    model = MODEL_REGISTRY[configs.model.model_name](configs=configs, tokenizer=tokenizer)
    model.build_model()
    optimizer, lr_scheduler = model.configure_optimizers()
    if args.amp:
        scaler = GradScaler()

    data_module = DATA_MODULE_REGISTRY[configs.dataset.dataset](configs)
    if dist.is_main_process():
        data_module.prepare_data()
    dist.barrier()
    data_module_kwargs = dict(tokenizer=tokenizer)
    if args.num_train_samples is not None:
        data_module_kwargs['num_train_samples'] = args.num_train_samples
    if args.num_eval_samples is not None:
        data_module_kwargs['num_eval_samples'] = args.num_eval_samples
    data_module.setup(**data_module_kwargs)

    dist.barrier()

    if args.num_train_samples is None:
        args.num_train_samples = len(data_module.dataset['train'])
    if args.num_eval_samples is None:
        args.num_eval_samples = len(data_module.dataset['val'])
    if args.eval_batch_size is None:
        args.eval_batch_size = args.batch_size

    train_dataloader = create_dataloader(
        data_module.dataset['train'],
        batch_size=args.batch_size,
        sampler_type='Distributed' if args.ddp else 'Random',
        worker_init_fn=worker_init)

    val_dataloader = create_dataloader(
        data_module.dataset['val'],
        batch_size=args.eval_batch_size,
        sampler_type='Distributed' if args.ddp else 'Sequential',
        worker_init_fn=worker_init)

    if args.max_steps is not None and args.max_steps > 0:
        max_epochs = args.max_steps // len(train_dataloader)
        args.max_epochs = max_epochs + 1 if (args.max_steps % len(train_dataloader)) \
                          else max_epochs
    else:
        assert(args.max_epochs > 0)
        args.max_steps = args.max_epochs * len(train_dataloader)

    args.num_eval_steps = len(val_dataloader)

    training_state = TrainingState(
        max_steps=args.max_steps,
        quality_target=args.quality_target,
        quality_judgement=args.quality_judgement)

    model.to(args.device)

    if args.ddp:
        model = DDP(model, device_ids=[args.local_rank])
        training_step = model.module.training_step
        validation_step = model.module.validation_step
    else:
        training_step = model.training_step
        validation_step = model.validation_step

    model.train()

    dist.barrier()
    train_start_time = time.perf_counter()
    train_time = 0
    start_time = train_start_time
    for epoch in range(1, args.max_epochs+1):
        
        if args.ddp:
            train_dataloader.sampler.set_epoch(epoch)
        
        train_data_iterator = iter(train_dataloader)
        for step in range(1, len(train_dataloader)+1):
            if training_state.end_training():
                break

            batch_data = next(train_data_iterator)
            training_state.global_step += 1

            if args.amp:
                with autocast():
                    batch_outputs = training_step(
                        batch=h2d_tensors(batch_data, args.device), batch_idx=step)
            else:
                batch_outputs = training_step(
                    batch=h2d_tensors(batch_data, args.device), batch_idx=step)

            optimizer.zero_grad()
            if args.amp:
                scaler.scale(batch_outputs['loss']).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                batch_outputs['loss'].backward()
                optimizer.step()
            loss_value = batch_outputs['loss']
            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)
            lr_scheduler.step()

            training_state.num_trained_samples += args.batch_size * args.num_gpus

            if training_state.global_step % args.log_freq == 0:
                dist.barrier()
                elapse = time.perf_counter() - start_time
                train_time += elapse
                if dist.is_main_process():
                    logger.log(
                        OrderedDict(
                            stage='train',
                            progress=OrderedDict(
                                epoch=epoch,
                                step=step),
                            metrics=OrderedDict(
                                loss=float(batch_outputs['loss'].detach().cpu()),
                                wer=batch_outputs['wer'],
                                cer=batch_outputs['cer']),
                            perf=OrderedDict(
                                tps=training_state.num_trained_samples / train_time)
                        )
                    )
                start_time = time.perf_counter()


            if training_state.global_step % args.eval_freq == 0:
                with torch.no_grad():
                    model.eval()
                    val_data_iterator = iter(val_dataloader)
                    metric_list = []
                    dist.barrier()
                    eval_start_time = time.perf_counter()
                    for eval_step in trange(1, args.num_eval_steps+1):
                        batch_data = next(val_data_iterator)
                        batch_outputs = validation_step(
                            batch=h2d_tensors(batch_data, args.device), batch_idx=eval_step)
                        metric_list.append(batch_outputs[args.quality_metric])

                    dist.barrier()
                    eval_duration = time.perf_counter() - eval_start_time
                    metric_value = np.mean(metric_list)
                    if args.ddp:
                        metric_value = np.mean(dist.all_gather(metric_value))
                        dist.barrier()
                    if dist.is_main_process():
                        logger.log(
                            OrderedDict(
                                stage='val',
                                progress=OrderedDict(
                                    epoch=epoch,
                                    step=args.num_eval_steps),
                                metrics={
                                    args.quality_metric: metric_value},
                                perf=OrderedDict(
                                    tps=args.num_eval_samples / eval_duration)
                            )
                        )
                    if training_state.meet_quality_target(metric_value):
                        training_state.status = training_state.Status.success
                        break

                model.train()
                start_time = time.perf_counter()

        if training_state.end_training():
            break

    dist.barrier()
    raw_train_time = time.perf_counter() - train_start_time
    final_state = OrderedDict(
        global_step=training_state.global_step,
        raw_train_time=raw_train_time,
        tps=training_state.num_trained_samples / raw_train_time,
        status={
            training_state.Status.success: 'success',
            training_state.Status.aborted: 'aborted'
        }[training_state.status])
    final_state[args.quality_metric] = metric_value

    if dist.is_main_process():
        logger.log(final_state)
    sys.exit(training_state.status)


if __name__ == '__main__':
    args = parse_args()
    try:
        from dltest import show_training_arguments
        show_training_arguments(args)
    except:
        pass
    main(args)
