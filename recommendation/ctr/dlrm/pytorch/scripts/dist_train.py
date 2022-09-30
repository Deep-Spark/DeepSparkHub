import datetime
import json
import itertools
import math
from pprint import pprint
from time import time

import numpy as np

from absl import app
from absl import flags
from absl import logging
import torch

from apex import amp
from apex import optimizers as apex_optim
from apex import parallel

from dlrm import dist_model
from dlrm.data import dataset
from dlrm.utils import distributed as dist
from dlrm.utils import metrics

import utils

from train import FLAGS

flags.DEFINE_string("backend", "nccl", "Backend to use for distributed training. Default nccl")
flags.DEFINE_boolean("cache_eval_data", False, "If True, cache eval data on first evaluation.")

def main(argv):
    if FLAGS.seed is not None:
        torch.manual_seed(FLAGS.seed)
        np.random.seed(FLAGS.seed)

    # Initialize distributed mode
    use_gpu = "cpu" not in FLAGS.device.lower()
    rank, world_size, gpu = dist.init_distributed_mode(backend=FLAGS.backend, use_gpu=use_gpu)
    if world_size == 1:
        raise NotImplementedError("This file is only for distributed training.")

    # Only print cmd args on rank 0
    if rank == 0:
        print("Command line flags:")
        pprint(FLAGS.flag_values_dict())

    # Check arguments sanity
    if FLAGS.batch_size % world_size != 0:
        raise ValueError(F"Batch size {FLAGS.batch_size} is not divisible by world_size {world_size}.")
    if FLAGS.test_batch_size % world_size != 0:
        raise ValueError(F"Test batch size {FLAGS.test_batch_size} is not divisible by world_size {world_size}.")

    # Load config file, create sub config for each rank
    with open(FLAGS.model_config, "r") as f:
        config = json.loads(f.read())

    wolrd_categorical_feature_sizes = np.asarray(config.pop('categorical_feature_sizes'))
    device_mapping = dist_model.get_criteo_device_mapping(world_size)
    vectors_per_gpu = device_mapping['vectors_per_gpu']
    # Get sizes of embeddings each GPU is gonna create
    categorical_feature_sizes = wolrd_categorical_feature_sizes[device_mapping['embedding'][rank]].tolist()

    bottom_mlp_sizes = config.pop('bottom_mlp_sizes')
    if rank != device_mapping['bottom_mlp']:
        bottom_mlp_sizes = None

    model = dist_model.DistDlrm(
        categorical_feature_sizes=categorical_feature_sizes,
        bottom_mlp_sizes=bottom_mlp_sizes,
        world_num_categorical_features=len(wolrd_categorical_feature_sizes),
        **config,
        device=FLAGS.device,
        use_embedding_ext=FLAGS.use_embedding_ext)

    dist.setup_distributed_print(rank == 0)

    # DDP introduces a gradient average through allreduce(mean), which doesn't apply to bottom model.
    # Compensate it with further scaling lr
    scaled_lr = FLAGS.lr / FLAGS.loss_scale if FLAGS.fp16 else FLAGS.lr
    scaled_lrs = [scaled_lr / world_size, scaled_lr]

    embedding_optimizer = torch.optim.SGD([
        {'params': model.bottom_model.joint_embedding.parameters(), 'lr': scaled_lrs[0]},
        ])
    mlp_optimizer = torch.optim.SGD([
        {'params': model.bottom_model.bottom_mlp.parameters(), 'lr': scaled_lrs[0]},
        {'params': model.top_model.parameters(), 'lr': scaled_lrs[1]}
        ])

    if FLAGS.fp16:
        (model.top_model, model.bottom_model.bottom_mlp), mlp_optimizer = amp.initialize(
            [model.top_model, model.bottom_model.bottom_mlp],
            mlp_optimizer, opt_level="O2", loss_scale=1, cast_model_outputs=torch.float16)

    if use_gpu:
        model.top_model = parallel.DistributedDataParallel(model.top_model)
    else:  # Use other backend for CPU
        model.top_model = torch.nn.parallel.DistributedDataParallel(model.top_model)

    loss_fn = torch.nn.BCEWithLogitsLoss(reduction="mean")

    # Too many arguments to pass for distributed training. Use plain train code here instead of
    # defining a train function

    # Print per 16384 * 2000 samples by default
    default_print_freq = 100
    print_freq = default_print_freq if FLAGS.print_freq is None else FLAGS.print_freq

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{avg:.4f}'))
    metric_logger.add_meter('step_time', utils.SmoothedValue(window_size=1, fmt='{avg:.4f} ms'))
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    # Accumulating loss on GPU to avoid memcpyD2H every step
    moving_loss = torch.zeros(1, device=FLAGS.device)
    moving_loss_stream = torch.cuda.Stream()

    local_embedding_device_mapping = torch.tensor(
        device_mapping['embedding'][rank], device=FLAGS.device, dtype=torch.long)

    # # LR is logged twice for now because of a compliance checker bug
    lr_scheduler = utils.LearningRateScheduler(optimizers=[mlp_optimizer, embedding_optimizer],
                                               base_lrs=[scaled_lrs, [scaled_lrs[0]]],
                                               warmup_steps=FLAGS.warmup_steps,
                                               warmup_factor=FLAGS.warmup_factor,
                                               decay_start_step=FLAGS.decay_start_step,
                                               decay_steps=FLAGS.decay_steps,
                                               decay_power=FLAGS.decay_power,
                                               end_lr_factor=FLAGS.decay_end_lr / FLAGS.lr)

    data_stream = torch.cuda.Stream()
    eval_data_cache = [] if FLAGS.cache_eval_data else None

    print("Creating data loaders")
    dist_dataset_args = {
        "numerical_features": rank == 0,
        "categorical_features": device_mapping['embedding'][rank]
    }

    data_loader_train, data_loader_test = dataset.get_data_loader(
        FLAGS.dataset, FLAGS.batch_size, FLAGS.test_batch_size, FLAGS.device,
        dataset_type=FLAGS.dataset_type, shuffle=FLAGS.shuffle,
        **dist_dataset_args)

    steps_per_epoch = len(data_loader_train)

    print(f"train dataloader length:{steps_per_epoch}")
    print(f"test dataloader length:{len(data_loader_test)}")

    # Default 20 tests per epoch
    test_freq = FLAGS.test_freq if FLAGS.test_freq is not None else steps_per_epoch // 20

    global_step = 0
    stop_time_print = time()
    start_time = time()
    stop_time = time()

    total_time = 0
    total_samples = 0
    samp_per_secs = []
    for epoch in range(FLAGS.epochs):
        step = 0
        epoch_start_time = time()

        for step, (numerical_features, categorical_features, click) in enumerate(
                dataset.prefetcher(iter(data_loader_train), data_stream)):
            t1 = time()
            torch.cuda.current_stream().wait_stream(data_stream)

            global_step = steps_per_epoch * epoch + step
            lr_scheduler.step()

            # Slice out categorical features if not using the "dist" dataset
            if FLAGS.dataset_type != "dist":
                categorical_features = categorical_features[:, local_embedding_device_mapping]

            if FLAGS.fp16 and categorical_features is not None:
                numerical_features = numerical_features.to(torch.float16)

            last_batch_size = None
            if click.shape[0] != FLAGS.batch_size: # last batch
                # for example,batchsize is 2048, 8 gpus, total sample number is 265, then gpu0 has 256(2048/8) samples, gpu1 has 9 samples,
                # and other gpus has none sample
                continue

            bottom_out = model.bottom_model(numerical_features, categorical_features)

            batch_size_per_gpu = FLAGS.batch_size // world_size
            from_bottom = dist_model.bottom_to_top(
                bottom_out, batch_size_per_gpu, config['embedding_dim'], vectors_per_gpu)

            if last_batch_size is not None:
               pass
            else:
                loss = loss_fn(
                    model.top_model(from_bottom).squeeze().float(),
                    click[rank * batch_size_per_gpu : (rank + 1) * batch_size_per_gpu])

            # We don't need to accumulate gradient. Set grad to None is faster than optimizer.zero_grad()
            for param_group in itertools.chain(embedding_optimizer.param_groups, mlp_optimizer.param_groups):
                for param in param_group['params']:
                    param.grad = None

            if FLAGS.fp16:
                loss *= FLAGS.loss_scale
                with amp.scale_loss(loss, mlp_optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            mlp_optimizer.step()
            embedding_optimizer.step()

            total_time += time() - t1
            total_samples += numerical_features.shape[0]*world_size

            if global_step == 0:
                print(F"Started epoch {epoch}...")
            elif global_step % print_freq == 0:
                torch.cuda.synchronize()
                # Averaging cross a print_freq period to reduce the error.
                # An accurate timing needs synchronize which would slow things down.
                metric_logger.update(
                    step_time=(time() - stop_time_print) * 1000 / print_freq,
                    loss=loss.item() / (FLAGS.loss_scale if FLAGS.fp16 else 1),
                    lr=mlp_optimizer.param_groups[1]["lr"] * (FLAGS.loss_scale if FLAGS.fp16 else 1))
                stop_time_print = time()
                samp_per_sec = total_samples / total_time
                samp_per_secs.append(samp_per_sec)
                metric_logger.print(
                    header=F"Epoch:[{epoch}/{FLAGS.epochs}] [{global_step}/{steps_per_epoch*FLAGS.epochs}]  speed: {format(samp_per_sec, '0.2f')+' records/s'}")
                moving_loss = 0.
                with torch.cuda.stream(moving_loss_stream):
                    moving_loss = 0.

            if global_step % test_freq == 0 and global_step > 0 and global_step / steps_per_epoch >= FLAGS.test_after:
                auc = dist_evaluate(model, data_loader_test, eval_data_cache)
                print(F"Epoch {epoch} step {global_step}. auc {auc:.6f}")
                stop_time = time()

                if auc > FLAGS.auc_threshold:
                    run_time_s = int(stop_time - start_time)
                    print(F"Hit target accuracy AUC {FLAGS.auc_threshold} at epoch "
                          F"{global_step/steps_per_epoch:.2f} in {run_time_s}s. "
                          F"Average training speed {np.array(samp_per_secs).mean():.1f} records/s.")
                    return


        epoch_stop_time = time()
        epoch_time_s = epoch_stop_time - epoch_start_time
        print(F"Finished epoch {epoch} in {datetime.timedelta(seconds=int(epoch_time_s))}. "
              F"Average training speed {np.array(samp_per_secs).mean():.1f} records/s.")

def dist_evaluate(model, data_loader, data_cache):
    """Test distributed DLRM model

    Args:
        model (DistDLRM):
        data_loader (torch.utils.data.DataLoader):
    """
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    device_mapping = dist_model.get_criteo_device_mapping(world_size)
    vectors_per_gpu = device_mapping['vectors_per_gpu']

    # Test batch size could be big, make sure it prints
    default_print_freq = max(16384 * 2000 // FLAGS.test_batch_size, 1)
    print_freq = default_print_freq if FLAGS.print_freq is None else FLAGS.print_freq

    steps_per_epoch = len(data_loader)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('step_time', utils.SmoothedValue(window_size=1, fmt='{avg:.4f} ms'))
    local_embedding_device_mapping = torch.tensor(
        device_mapping['embedding'][rank], device=FLAGS.device, dtype=torch.long)
    with torch.no_grad():
        # ROC can be computed per batch and then compute AUC globally, but I don't have the code.
        # So pack all the outputs and labels together to compute AUC. y_true and y_score naming follows sklearn
        y_true = []
        y_score = []
        data_stream = torch.cuda.Stream()
        stop_time = time()

        if data_cache is None or not data_cache:
            eval_data_iter = dataset.prefetcher(iter(data_loader), data_stream)
        else:
            print("Use cached eval data")
            eval_data_iter = data_cache
        for step, (numerical_features, categorical_features, click) in enumerate(eval_data_iter):
            if data_cache is not None and len(data_cache) < steps_per_epoch:
                data_cache.append((numerical_features, categorical_features, click))
            last_batch_size = None
            if click.shape[0] != FLAGS.test_batch_size: # last batch
                last_batch_size = click.shape[0]
                logging.debug("Pad the last test batch of size %d to %d", last_batch_size, FLAGS.test_batch_size)
                padding_size = FLAGS.test_batch_size - last_batch_size
                padding_numiercal = torch.empty(
                    padding_size, numerical_features.shape[1],
                    device=numerical_features.device, dtype=numerical_features.dtype)
                numerical_features = torch.cat((numerical_features, padding_numiercal), dim=0)
                if categorical_features is not None:
                    padding_categorical = torch.ones(
                        padding_size, categorical_features.shape[1],
                        device=categorical_features.device, dtype=categorical_features.dtype)
                    categorical_features = torch.cat((categorical_features, padding_categorical), dim=0)

            if FLAGS.dataset_type != "dist":
                categorical_features = categorical_features[:, local_embedding_device_mapping]

            if FLAGS.fp16 and categorical_features is not None:
                numerical_features = numerical_features.to(torch.float16)
            bottom_out = model.bottom_model(numerical_features, categorical_features)
            batch_size_per_gpu = FLAGS.test_batch_size // world_size
            from_bottom = dist_model.bottom_to_top(bottom_out, batch_size_per_gpu, model.embedding_dim, vectors_per_gpu)

            output = model.top_model(from_bottom).squeeze()

            buffer_dtype = torch.float32 if not FLAGS.fp16 else torch.float16
            output_receive_buffer = torch.empty(FLAGS.test_batch_size, device=FLAGS.device, dtype=buffer_dtype)
            torch.distributed.all_gather(list(output_receive_buffer.split(batch_size_per_gpu)), output)
            if last_batch_size is not None:
                output_receive_buffer = output_receive_buffer[:last_batch_size]

            y_true.append(click)
            y_score.append(output_receive_buffer.float())

            if step % print_freq == 0 and step != 0:
                torch.cuda.synchronize()
                metric_logger.update(step_time=(time() - stop_time) * 1000 / print_freq)
                stop_time = time()
                metric_logger.print(header=F"Test: [{step}/{steps_per_epoch}]")

        auc = metrics.roc_auc_score(torch.cat(y_true), torch.sigmoid(torch.cat(y_score).float()))

    return auc

if __name__ == '__main__':
    app.run(main)
