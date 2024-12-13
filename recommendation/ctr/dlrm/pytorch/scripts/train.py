"""Reference training script
Only Criteo data is supported at the moment, one hot embedding.
"""
import datetime
import functools
import itertools
import json
import os
import sys
import math
from pprint import pprint
from time import time

from absl import app
from absl import flags

import torch
from apex import amp

from dlrm import dist_model
from dlrm.data import dataset
from dlrm.utils import metrics

import utils

FLAGS = flags.FLAGS

# Basic run settings
flags.DEFINE_integer("seed", None, "Random seed")
flags.DEFINE_boolean("test_only", False, "")
flags.DEFINE_integer("profile_steps", None, "If set, quit on profile_steps")

# Training schedule flags
flags.DEFINE_integer("batch_size", 16384, "")
flags.DEFINE_integer("test_batch_size", 131072, "Test can use much larger batch size")
flags.DEFINE_float("lr", 8.0, "Base learning rate")
flags.DEFINE_integer("epochs", 1, "Number of epochs to train.")
flags.DEFINE_integer("warmup_factor", 0, "Learning rate warmup factor. Must be a non-negative integer")
flags.DEFINE_integer("warmup_steps", 1000, "Number of warmup optimization steps")
flags.DEFINE_integer("decay_steps", 0, "Polynomial learning rate decay steps. If equal to 0 will not do any decaying")
flags.DEFINE_integer(
    "decay_start_step", 48000,
    ("Optimization step after which to start decaying the learning rate,"
     "if None will start decaying right after the warmup phase is completed"))
flags.DEFINE_integer("decay_power", 2, "Polynomial learning rate decay power")
flags.DEFINE_float("decay_end_lr", 8., "LR after the decay ends")

# Model configuration
flags.DEFINE_string("model_config", None, "json file of model configuration")
flags.DEFINE_string(
    "dataset", None,
    "Full path to reference binary dataset. Must have filename, train_data.bin ... day_fea_count.npz")
flags.DEFINE_boolean("shuffle_batch_order", True, "Read batch in train dataset by random order", short_name="shuffle")
flags.DEFINE_enum("dataset_type", "memmap", ["bin", "memmap", "dist"], "Which dataset to use.")
flags.DEFINE_boolean("use_embedding_ext", True, "Use embedding cuda extension. If False, use Pytorch embedding")

# Saving and logging flags
flags.DEFINE_string("output_dir", "/tmp", "path where to save")
flags.DEFINE_integer("test_freq", None, "#steps test. If None, 20 tests per epoch per MLperf rule.")
flags.DEFINE_float("test_after", 0, "Don't test the model unless this many epochs has been completed")
flags.DEFINE_integer("print_freq", None, "#steps per pring")
flags.DEFINE_string("ckpt", None, "load from checkpoint")
flags.DEFINE_boolean("save_model", False, "Save trained model")

# Machine setting flags
flags.DEFINE_enum("device", "cuda", ["cuda", "cpu"], "Device to run the model.")
flags.DEFINE_boolean("fp16", True, "Use fp16")
flags.DEFINE_float("loss_scale", 1024, "Static loss scale for fp16")

#Accuracy flags
flags.DEFINE_float("auc_threshold", 0.8025, "Target AUC.")

flags.mark_flags_as_required(["dataset", "model_config"])

def main(argv):
    try:
        from dltest import show_training_arguments
        show_training_arguments(FLAGS)
    except:
        pass
    if FLAGS.seed is not None:
        torch.manual_seed(FLAGS.seed)

    print("Command line flags:")
    pprint(FLAGS.flag_values_dict())

    print("Creating data loaders")
    FLAGS.test_batch_size =  FLAGS.batch_size
    data_loader_train, data_loader_test = dataset.get_data_loader(
        FLAGS.dataset, FLAGS.batch_size, FLAGS.test_batch_size, FLAGS.device,
        dataset_type=FLAGS.dataset_type, shuffle=FLAGS.shuffle)

    print("Creating model")
    with open(FLAGS.model_config, "r") as f:
        config = json.loads(f.read())
    model = dist_model.DistDlrm(
        **config,
        world_num_categorical_features=len(config['categorical_feature_sizes']),
        device=FLAGS.device)

    loss_fn = torch.nn.BCEWithLogitsLoss(reduction="mean")

    model.to(FLAGS.device)

    scaled_lr = FLAGS.lr / FLAGS.loss_scale if FLAGS.fp16 else FLAGS.lr
    optimizer = torch.optim.SGD(itertools.chain(
        model.bottom_model.parameters(), model.top_model.parameters()), lr=scaled_lr)

    if FLAGS.fp16:
        # Set loss scale to 1 for amp to bypass inf/nan check in amp that doesn't support sparse tensor used
        # in nn.Embedding. Loss scale will be cancelled by scaled learning rate
        (model.top_model, model.bottom_model.bottom_mlp), optimizer = amp.initialize(
            [model.top_model, model.bottom_model.bottom_mlp], optimizer, opt_level="O2", loss_scale=1)

    if FLAGS.test_only:
        if FLAGS.ckpt is None:
            raise ValueError(F"Checkpoint must be supplied when test_only is True.")
        loss, auc = evaluate(model, loss_fn, data_loader_test)
        print(F"Finished testing. Test Loss {loss:.4f}, auc {auc:.4f}")
        return

    train(model, loss_fn, optimizer, data_loader_train, data_loader_test, scaled_lr)

    # Saving such big model is slow and disk consuming. We don't need to save the model most of the time.
    # Make it optional and default to not save.
    if FLAGS.save_model:
        save_path = F"model_{FLAGS.epochs}.pt"
        if FLAGS.output_dir is not None:
            if not os.path.exists(FLAGS.output_dir):
                print(F"Creating {FLAGS.output_dir}")
                os.mkdir(FLAGS.output_dir)
            save_path = os.path.join(FLAGS.output_dir, save_path)
        torch.save(model.state_dict(), save_path)

def train(model, loss_fn, optimizer, data_loader_train, data_loader_test, scaled_lr):
    """Train and evaluate the model

    Args:
        model (dlrm):
        loss_fn (torch.nn.Module): Loss function
        optimizer (torch.nn.optim):
        data_loader_train (torch.utils.data.DataLoader):
        data_loader_test (torch.utils.data.DataLoader):
        scaled_lr (float)
    """
    # Print per 16384 * 2000 samples by default
    default_print_freq = 16384 * 2000 // FLAGS.batch_size
    print_freq = default_print_freq if FLAGS.print_freq is None else FLAGS.print_freq

    steps_per_epoch = len(data_loader_train)
    # MLperf requires 20 tests per epoch
    test_freq = FLAGS.test_freq if FLAGS.test_freq is not None else steps_per_epoch // 20

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=print_freq, fmt='{avg:.4f}'))
    metric_logger.add_meter('step_time', utils.SmoothedValue(window_size=1, fmt='{avg:.4f}'))
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))

    lr_scheduler = utils.LearningRateScheduler(optimizers=[optimizer],
                                               base_lrs=[scaled_lr],
                                               warmup_steps=FLAGS.warmup_steps,
                                               warmup_factor=FLAGS.warmup_factor,
                                               decay_start_step=FLAGS.decay_start_step,
                                               decay_steps=FLAGS.decay_steps,
                                               decay_power=FLAGS.decay_power,
                                               end_lr_factor=FLAGS.decay_end_lr / FLAGS.lr)

    step = 0
    start_time = time()
    stop_time = time()
    for epoch in range(FLAGS.epochs):
        epoch_start_time = time()

        for numerical_features, categorical_features, click in data_loader_train:
            global_step = steps_per_epoch * epoch + step
            lr_scheduler.step()

            output = model(numerical_features, categorical_features).squeeze()
            loss = loss_fn(output, click)

            optimizer.zero_grad()
            if FLAGS.fp16:
                loss *= FLAGS.loss_scale
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            optimizer.step()
            loss_value = loss.item()
            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)
            # Cancel loss scale for logging if fp16 is used
            metric_logger.update(
                loss=loss.item() / (FLAGS.loss_scale if FLAGS.fp16 else 1),
                lr=optimizer.param_groups[0]["lr"] * (FLAGS.loss_scale if FLAGS.fp16 else 1))

            if step % print_freq == 0:
                # Averaging cross a print_freq period to reduce the error.
                # An accurate timing needs synchronize which would slow things down.
                metric_logger.update(step_time=(time() - stop_time) / print_freq)
                stop_time = time()
                eta_str = datetime.timedelta(seconds=int(metric_logger.step_time.global_avg * (steps_per_epoch - step)))
                metric_logger.print(
                    header=F"Epoch:[{epoch}/{FLAGS.epochs}] [{step}/{steps_per_epoch}]  eta: {eta_str}")

            if global_step % test_freq == 0 and global_step > 0 and global_step / steps_per_epoch >= FLAGS.test_after:
                loss, auc = evaluate(model, loss_fn, data_loader_test)
                if not math.isfinite(loss):
                    print("Loss is {}, stopping training".format(loss))
                    sys.exit(1)
                if not math.isfinite(auc):
                    print("AUC is {}, stopping training".format(auc))
                    sys.exit(1)
                print(F"Epoch {epoch} step {step}. Test loss {loss:.4f}, auc {auc:.6f}")
                stop_time = time()

                if auc >= FLAGS.auc_threshold:
                    run_time_s = int(stop_time - start_time)
                    print(F"Hit target accuracy AUC {FLAGS.auc_threshold} at epoch "
                          F"{global_step/steps_per_epoch:.2f} in {run_time_s}s. "
                          F"Average speed {global_step * FLAGS.batch_size / run_time_s:.1f} records/s.")
                    return
            step += 1

        epoch_stop_time = time()
        epoch_time_s = epoch_stop_time - epoch_start_time
        print(F"Finished epoch {epoch} in {datetime.timedelta(seconds=int(epoch_time_s))}. "
              F"Average speed {steps_per_epoch * FLAGS.batch_size / epoch_time_s:.1f} records/s.")


def evaluate(model, loss_fn, data_loader):
    """Test dlrm model

    Args:
        model (dlrm):
        loss_fn (torch.nn.Module): Loss function
        data_loader (torch.utils.data.DataLoader):
    """
    # Test bach size could be big, make sure it prints
    default_print_freq = max(524288 * 100 // FLAGS.test_batch_size, 1)
    print_freq = default_print_freq if FLAGS.print_freq is None else FLAGS.print_freq

    steps_per_epoch = len(data_loader)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=print_freq, fmt='{avg:.4f}'))
    metric_logger.add_meter('step_time', utils.SmoothedValue(window_size=1, fmt='{avg:.4f}'))
    with torch.no_grad():
        # ROC can be computed per batch and then compute AUC globally, but I don't have the code.
        # So pack all the outputs and labels together to compute AUC. y_true and y_score naming follows sklearn
        y_true = []
        y_score = []
        stop_time = time()
        for step, (numerical_features, categorical_features, click) in enumerate(data_loader):
            output = model(numerical_features, categorical_features).squeeze()
            loss = loss_fn(output, click)
            y_true.append(click)
            y_score.append(output)

            metric_logger.update(loss=loss.item())
            if step % print_freq == 0:
                metric_logger.update(step_time=(time() - stop_time) / print_freq)
                stop_time = time()
                metric_logger.print(header=F"Test: [{step}/{steps_per_epoch}]")

        auc = metrics.roc_auc_score(torch.cat(y_true), torch.sigmoid(torch.cat(y_score)))

    return metric_logger.loss.global_avg, auc

if __name__ == '__main__':
    app.run(main)
