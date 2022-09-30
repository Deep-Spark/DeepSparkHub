#! /usr/bin/env python3

from __future__ import division

import os
import argparse
import tqdm
import time
import datetime
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../..'))

import torch

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    class SummaryWriter(object):
        def __init__(self, log_dir=None, comment='', purge_step=None, max_queue=10,
                     flush_secs=120, filename_suffix=''):
            if not log_dir:
                import socket
                from datetime import datetime
                current_time = datetime.now().strftime('%b%d_%H-%M-%S')
                log_dir = os.path.join(
                    'runs', current_time + '_' + socket.gethostname() + comment)
            self.log_dir = log_dir
            self.purge_step = purge_step
            self.max_queue = max_queue
            self.flush_secs = flush_secs
            self.filename_suffix = filename_suffix

            # Initialize the file writers, but they can be cleared out on close
            # and recreated later as needed.
            self.file_writer = self.all_writers = None
            self._get_file_writer()

            # Create default bins for histograms, see generate_testdata.py in tensorflow/tensorboard
            v = 1E-12
            buckets = []
            neg_buckets = []
            while v < 1E20:
                buckets.append(v)
                neg_buckets.append(-v)
                v *= 1.1
            self.default_bins = neg_buckets[::-1] + [0] + buckets

        def _check_caffe2_blob(self, item): pass

        def _get_file_writer(self): pass

        def get_logdir(self):
            """Returns the directory where event files will be written."""
            return self.log_dir

        def add_hparams(self, hparam_dict, metric_dict, hparam_domain_discrete=None, run_name=None): pass

        def add_scalar(self, tag, scalar_value, global_step=None, walltime=None, new_style=False): pass

        def add_scalars(self, main_tag, tag_scalar_dict, global_step=None, walltime=None): pass

        def add_histogram(self, tag, values, global_step=None, bins='tensorflow', walltime=None, max_bins=None): pass

        def add_histogram_raw(self, tag, min, max, num, sum, sum_squares, bucket_limits, bucket_counts, global_step=None, walltime=None): pass

        def add_image(self, tag, img_tensor, global_step=None, walltime=None, dataformats='CHW'): pass

        def add_images(self, tag, img_tensor, global_step=None, walltime=None, dataformats='NCHW'): pass

        def add_image_with_boxes(self, tag, img_tensor, box_tensor, global_step=None, walltime=None, rescale=1, dataformats='CHW', labels=None): pass

        def add_figure(self, tag, figure, global_step=None, close=True, walltime=None): pass

        def add_video(self, tag, vid_tensor, global_step=None, fps=4, walltime=None): pass

        def add_audio(self, tag, snd_tensor, global_step=None, sample_rate=44100, walltime=None): pass

        def add_text(self, tag, text_string, global_step=None, walltime=None): pass

        def add_onnx_graph(self, prototxt): pass

        def add_graph(self, model, input_to_model=None, verbose=False): pass

        @staticmethod
        def _encode(rawstr): pass

        def add_embedding(self, mat, metadata=None, label_img=None, global_step=None, tag='default', metadata_header=None): pass

        def add_pr_curve(self, tag, labels, predictions, global_step=None, num_thresholds=127, weights=None, walltime=None): pass

        def add_pr_curve_raw(self, tag, true_positive_counts, false_positive_counts, true_negative_counts, false_negative_counts, precision, recall, global_step=None, num_thresholds=127, weights=None, walltime=None): pass

        def add_custom_scalars_multilinechart(self, tags, category='default', title='untitled'): pass

        def add_custom_scalars_marginchart(self, tags, category='default', title='untitled'): pass

        def add_custom_scalars(self, layout): pass

        def add_mesh(self, tag, vertices, colors=None, faces=None, config_dict=None, global_step=None, walltime=None): pass

        def flush(self): pass

        def close(self): pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.close()


from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim as optim
from torch.cuda import amp

from pytorchyolo.models import load_model
from pytorchyolo.utils.logger import Logger
from pytorchyolo.utils.utils import to_cpu, load_classes, print_environment_info, worker_seed_set
from pytorchyolo.utils.datasets import ListDataset
from pytorchyolo.utils.augmentations import AUGMENTATION_TRANSFORMS
# from pytorchyolo.utils.transforms import DEFAULT_TRANSFORMS
from pytorchyolo.utils.parse_config import parse_data_config
from pytorchyolo.utils.loss import compute_loss
from pytorchyolo.test import _evaluate, _create_validation_data_loader

from terminaltables import AsciiTable

from torchsummary import summary
from common_utils import init_distributed_mode



train_names = ["module_list.81.conv_81.weight", "module_list.81.conv_81.bias", 
                "module_list.93.conv_93.weight", "module_list.93.conv_93.bias",
                "module_list.105.conv_105.weight", "module_list.105.conv_105.bias"]

def _create_data_loader(img_path, batch_size, img_size, n_cpu, multiscale_training=False, distributed=False):
    """Creates a DataLoader for training.

    :param img_path: Path to file containing all paths to training images.
    :type img_path: str
    :param batch_size: Size of each image batch
    :type batch_size: int
    :param img_size: Size of each image dimension for yolo
    :type img_size: int
    :param n_cpu: Number of cpu threads to use during batch generation
    :type n_cpu: int
    :param multiscale_training: Scale images to different sizes randomly
    :type multiscale_training: bool
    :return: Returns DataLoader
    :rtype: DataLoader
    """
    dataset = ListDataset(
        img_path,
        img_size=img_size,
        multiscale=multiscale_training,
        transform=AUGMENTATION_TRANSFORMS)
    sampler = None
    shuffle = True
    if distributed:
        sampler = DistributedSampler(dataset, rank=dist.get_rank(), shuffle=True)
        shuffle = False
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
        worker_init_fn=worker_seed_set,
        sampler=sampler
    )
    return dataloader


def run():
    print_environment_info()
    start_time = time.time()
    parser = argparse.ArgumentParser(description="Trains the YOLO model.")
    parser.add_argument("-m", "--model", type=str, default="config/yolov3-voc.cfg", help="Path to model definition file (.cfg)")
    parser.add_argument("-d", "--data", type=str, default="config/voc.data", help="Path to data config file (.data)")
    parser.add_argument("-e", "--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("-v", "--verbose", action='store_true', help="Makes the training more verbose")
    parser.add_argument("--n_cpu", type=int, default=0, help="Number of cpu threads to use during batch generation")
    parser.add_argument("--pretrained_weights", type=str, help="Path to checkpoint file (.weights or .pth). Starts training from checkpoint model")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="Interval of epochs between saving model weights")
    parser.add_argument("--evaluation_interval", type=int, default=1, help="Interval of epochs between evaluations on validation set")
    parser.add_argument("--multiscale_training", action="store_false", help="Allow for multi-scale training")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="Evaluation: IOU threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.01, help="Evaluation: Object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="Evaluation: IOU threshold for non-maximum suppression")
    parser.add_argument("--logdir", type=str, default="logs", help="Directory for training log files (e.g. for TensorBoard)")
    parser.add_argument("--second_stage_steps", type=int, default=10, help="Number of second stage training steps(unfreeze all params)")

    # distributed training parameters
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='Local rank')
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    parser.add_argument("--dist_backend", type=str, default="gloo", help="Distributed training backend.")

    parser.add_argument('--amp', action='store_true', default=False, help='use amp to train and test')
    args = parser.parse_args()

    args.rank = -1
    init_distributed_mode(args)
    rank = args.rank

    print(f"Command line arguments: {args}")

    logger = Logger(args.logdir)  # Tensorboard logger

    # Create output directories if missing
    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # enable cudnn autotune
    torch.backends.cudnn.benchmark = True

    # Get data configuration
    data_config = parse_data_config(args.data)
    train_path = data_config["train"]
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ############
    # Create model
    # ############

    model = load_model(args.model, args.pretrained_weights)
    model_module = model
    if args.distributed:
        model = model.to(rank)
        model = DDP(model, device_ids=[args.rank], find_unused_parameters=True)
        model_module = model.module

    # Print model
    if args.verbose:
        summary(model_module, input_size=(3, model_module.hyperparams['height'], model_module.hyperparams['height']))

    mini_batch_size = model_module.hyperparams['batch'] // model_module.hyperparams['subdivisions']

    if dist.is_initialized():
        if dist.get_world_size() >= 8:
            _origin_bs = mini_batch_size
            mini_batch_size = mini_batch_size // 4
            mini_batch_size = max(4, mini_batch_size)
            print(f"WARN: Updating batch size from {_origin_bs} to {mini_batch_size} in per process, avoid non-convergence when training small dataset.")

    # #################
    # Create Dataloader
    # #################

    # Load training dataloader
    dataloader = _create_data_loader(
        train_path,
        mini_batch_size,
        model_module.hyperparams['height'],
        args.n_cpu,
        args.multiscale_training,
        distributed=args.distributed
    )

    # Load validation dataloader
    validation_dataloader = _create_validation_data_loader(
        valid_path,
        mini_batch_size,
        model_module.hyperparams['height'],
        args.n_cpu
    )

    # ################
    # Create optimizer
    # ################

    params = [p for p in model.parameters() if p.requires_grad]
    print("===== Print trainable parameters =====")
    print("Number of all parameters is {}".format(len(list(model.parameters())))) # 222
    # Should not print anything
    for name, param in model.named_parameters():
        if not param.requires_grad:
            print(name, param.data.shape)
    
    # Freeze backbone network params
    other_names = []
    for name, param in model.named_parameters():
        if rank != -1 and name.startswith('module.'):
            # DDP
            name = name[len('module.'):]
        if name in train_names:
            print(name, param.data.shape)
        else:
            param.requires_grad = False
            other_names.append(name)
    params = [p for p in model.parameters() if p.requires_grad]
    
    if (model_module.hyperparams['optimizer'] in [None, "adam"]):
        optimizer = optim.Adam(
            params,
            lr=model_module.hyperparams['learning_rate'],
            weight_decay=model_module.hyperparams['decay'],
        )
    elif (model_module.hyperparams['optimizer'] == "sgd"):
        optimizer = optim.SGD(
            params,
            lr=model_module.hyperparams['learning_rate'],
            weight_decay=model_module.hyperparams['decay'],
            momentum=model_module.hyperparams['momentum'])
    else:
        print("Unknown optimizer. Please choose between (adam, sgd).")

    scaler = amp.GradScaler()

    checkpoint_path = None
    # First stage training
    for epoch in range(args.epochs):

        print("\n---- Finetuning Model ----")
        epoch_start_time = time.time()

        model.train()  # Set model to training mode

        for batch_i, (img_paths, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc=f"Training Epoch {epoch}")):
            batches_done = len(dataloader) * epoch + batch_i

            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device)

            # print("len of img_paths = {}".format(len(img_paths)))
            # print(img_paths)
            # print(len(targets))
            # for target in targets:
            #     print("{}\t|\t{}".format(target, target.shape))
            # print(1/0)
            if args.amp:
                with amp.autocast():
                    outputs = model(imgs)
                    loss, loss_components = compute_loss(outputs, targets, model_module)
                    scaler.scale(loss).backward()
            else:
                outputs = model(imgs)

                loss, loss_components = compute_loss(outputs, targets, model_module)

                loss.backward()

            ###############
            # Run optimizer
            ###############

            if batches_done % model_module.hyperparams['subdivisions'] == 0:
                # Adapt learning rate
                # Get learning rate defined in cfg
                lr = model_module.hyperparams['learning_rate']
                if batches_done < model_module.hyperparams['burn_in']:
                    # Burn in
                    lr *= (batches_done / model_module.hyperparams['burn_in'])
                else:
                    # Set and parse the learning rate to the steps defined in the cfg
                    for threshold, value in model_module.hyperparams['lr_steps']:
                        if batches_done > threshold:
                            lr *= value
                # Log the learning rate
                if rank in [-1, 0]:
                    logger.scalar_summary("train/learning_rate", lr, batches_done)
                # Set learning rate
                for g in optimizer.param_groups:
                    g['lr'] = lr

                # Run optimizer
                if args.amp:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                # Reset gradients
                optimizer.zero_grad()

            # ############
            # Log progress
            # ############
            if args.verbose and rank in [-1, 0]:
                print(AsciiTable(
                    [
                        ["Type", "Value"],
                        ["IoU loss", float(loss_components[0])],
                        ["Object loss", float(loss_components[1])],
                        ["Class loss", float(loss_components[2])],
                        ["Loss", float(loss_components[3])],
                        ["Batch loss", to_cpu(loss).item()],
                    ]).table)

            # Tensorboard logging
            if rank in [-1, 0]:
                tensorboard_log = [
                    ("train/iou_loss", float(loss_components[0])),
                    ("train/obj_loss", float(loss_components[1])),
                    ("train/class_loss", float(loss_components[2])),
                    ("train/loss", to_cpu(loss).item())]
                logger.list_of_scalars_summary(tensorboard_log, batches_done)

            model_module.seen += imgs.size(0)

        # #############
        # Save progress
        # #############

        # Save model to checkpoint file
        if (epoch % args.checkpoint_interval == 0) and (rank in [-1, 0]):
            checkpoint_path = f"checkpoints/yolov3_ckpt_{epoch}.pth"
            print(f"---- Saving checkpoint to: '{checkpoint_path}' ----")
            saved_state_dict = {}
            _state_dict = model.state_dict()
            for k in _state_dict.keys():
                new_k = k
                if k.startswith('module.'):
                    new_k = k[len('module.'):]
                saved_state_dict[new_k] = _state_dict[k]
            torch.save(saved_state_dict, checkpoint_path)
        epoch_total_time = time.time() - epoch_start_time
        epoch_total_time_str = str(datetime.timedelta(seconds=int(epoch_total_time)))

        fps = len(dataloader) * mini_batch_size / epoch_total_time
        if dist.is_initialized():
            fps = fps * dist.get_world_size()

        print('epoch time {}, Total FPS: {}'.format(epoch_total_time_str, fps))

    # Unfreeze all params
    # if (checkpoint_path is not None) and (rank == -1):
    #     print('Load checkpoint')
    #     model = load_model(args.model, checkpoint_path) # Why do we need to restore?
    #     for name, param in model.named_parameters():
    #         param.requires_grad = True
    #
    #     model_module = model
    #     if args.distributed:
    #         model = DDP(model, device_ids=[args.rank])
    #         model_module = model.module
    #
    # print('Resume training')
    # # other_params = []
    # # for name, param in model.named_parameters():
    # #     if name in other_names:
    # #         param.requires_grad = True
    # #         other_params.append(param)
    #
    # # optimizer.param_groups.append({'params': other_params})
    # # params = [p for p in model.parameters() if p.requires_grad]
    # # Reset optimizer
    # optimizer.zero_grad()
    # if torch.cuda.is_available():
    #     model.module.cuda()
    # model.train()
    # model_module.train()
    # print(
    #     'model', type(model), '\n',
    #     'model module', type(model_module)
    # )
    # for name, param in model.named_parameters():
    #     param.requires_grad = True
    # params = model.parameters()
    # if (model_module.hyperparams['optimizer'] in [None, "adam"]):
    #     optimizer = optim.Adam(
    #         params,
    #         lr=lr,
    #         weight_decay=model_module.hyperparams['decay'],
    #     )
    # elif (model_module.hyperparams['optimizer'] == "sgd"):
    #     optimizer = optim.SGD(
    #         params,
    #         lr=lr,
    #         weight_decay=model_module.hyperparams['decay'],
    #         momentum=model_module.hyperparams['momentum'])
    # else:
    #     print("Unknown optimizer. Please choose between (adam, sgd).")

    # # Second stage training
    # epoch += 1
    # dist.barrier()
    # dataloader_iter = iter(dataloader)
    # for batch_i in tqdm.tqdm(range(args.second_stage_steps), desc=f"Training Epoch {epoch}"):
    #     (img_paths, imgs, targets) = next(dataloader_iter)
    # # for batch_i, (img_paths, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc=f"Training Epoch {epoch}")):
    #     batches_done = len(dataloader) * epoch + batch_i
    #
    #     imgs = imgs.to(device, non_blocking=True)
    #     targets = targets.to(device)
    #
    #     outputs = model(imgs)
    #
    #     loss, loss_components = compute_loss(outputs, targets, model_module)
    #
    #     loss.backward()
    #
    #     ###############
    #     # Run optimizer
    #     ###############
    #
    #     if batches_done % model_module.hyperparams['subdivisions'] == 0:
    #         # Adapt learning rate
    #         # Get learning rate defined in cfg
    #         lr = model_module.hyperparams['learning_rate']
    #         if batches_done < model_module.hyperparams['burn_in']:
    #             # Burn in
    #             lr *= (batches_done / model_module.hyperparams['burn_in'])
    #         else:
    #             # Set and parse the learning rate to the steps defined in the cfg
    #             for threshold, value in model_module.hyperparams['lr_steps']:
    #                 if batches_done > threshold:
    #                     lr *= value
    #         # Log the learning rate
    #         logger.scalar_summary("train/learning_rate", lr, batches_done)
    #         # Set learning rate
    #         for g in optimizer.param_groups:
    #             g['lr'] = lr
    #         g['lr'] = 3e-7
    #         # Run optimizer
    #         optimizer.step()
    #         # Reset gradients
    #         optimizer.zero_grad()
    #
    #     # ############
    #     # Log progress
    #     # ############
    #     if args.verbose:
    #         print(AsciiTable(
    #             [
    #                 ["Type", "Value"],
    #                 ["IoU loss", float(loss_components[0])],
    #                 ["Object loss", float(loss_components[1])],
    #                 ["Class loss", float(loss_components[2])],
    #                 ["Loss", float(loss_components[3])],
    #                 ["Batch loss", to_cpu(loss).item()],
    #             ]).table)
    #
    #     # Tensorboard logging
    #     tensorboard_log = [
    #         ("train/iou_loss", float(loss_components[0])),
    #         ("train/obj_loss", float(loss_components[1])),
    #         ("train/class_loss", float(loss_components[2])),
    #         ("train/loss", to_cpu(loss).item())]
    #     logger.list_of_scalars_summary(tensorboard_log, batches_done)
    #
    #     model_module.seen += imgs.size(0)

    # #############
    # Save progress
    # #############

    # # Save model to checkpoint file
    # if epoch % args.checkpoint_interval == 0:
    #     checkpoint_path = f"checkpoints/yolov3_ckpt_{epoch}.pth"
    #     print(f"---- Saving checkpoint to: '{checkpoint_path}' ----")
    #     torch.save(model_module.state_dict(), checkpoint_path)

    # ########
    # Evaluate
    # ########

    print("\n---- Evaluating Model ----")
    # Evaluate the model on the validation set
    metrics_output = _evaluate(
        model_module,
        validation_dataloader,
        class_names,
        img_size=model_module.hyperparams['height'],
        iou_thres=args.iou_thres,
        conf_thres=args.conf_thres,
        nms_thres=args.nms_thres,
        verbose=True
    )

    if (metrics_output is not None) and (rank in [-1, 0]):
        precision, recall, AP, f1, ap_class = metrics_output
        evaluation_metrics = [
            ("validation/precision", precision.mean()),
            ("validation/recall", recall.mean()),
            ("validation/mAP", AP.mean()),
            ("validation/f1", f1.mean())]
        logger.list_of_scalars_summary(evaluation_metrics, epoch)
        with open("train.logs", 'a') as f:
            f.write("epoch = {}\n".format(epoch))
            f.write("mAP = {}\n".format(AP.mean()))
            f.write("AP = \n")
            for elem in AP:
                f.write("{}\n".format(elem))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == "__main__":
    run()
