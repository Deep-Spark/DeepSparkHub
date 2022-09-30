import torch
import os

from functools import partial

from torch.utils.data import DataLoader

from box_coder import dboxes300_coco
from .util import COCODetection, SSDTransformer
from .sampler import GeneralDistributedSampler

from pycocotools.coco import COCO
import numpy as np


def SSDCollator(batch, dali):
    """
    :param batch: batch data, [image, image_id, (htot, wtot), bboxes, labels]
        if Dali is False:
            a batch is like:
                [
                    [torch.Size([3, 300, 300]), 152915, (262, 386), torch.Size([8732, 4]), torch.Size([8732])],
                    [torch.Size([3, 300, 300]), 260111, (480, 640), torch.Size([8732, 4]), torch.Size([8732])]
                    ....
                ]
        if Dali is True:
            This function will not be called.
    :param dali: whether use Dali
    :return:
    """
    images = []
    image_ids = []
    image_sizes = []
    bboxes = []
    bbox_offsets = [0]
    labels = []
    for img, img_id, img_size, bbox, label in batch:
        images.append(img.view(1, *img.shape))
        image_ids.append(img_id)
        image_sizes.append(img_size)
        bboxes.append(bbox)
        labels.append(label)
        bbox_offsets.append(bbox_offsets[-1] + bbox.shape[0])

    images = torch.cat(images)
    bboxes = torch.cat(bboxes)
    labels = torch.cat(labels)
    if dali:
        bbox_offsets = np.array(bbox_offsets).astype(np.int32)
        return [images, bboxes, labels, torch.tensor(bbox_offsets)]
    else:
        N = images.shape[0]
        return [images, bboxes.view(N, -1, 4), labels.view(N, -1)]


def build_train_pipe(config):
    input_size = config.input_size
    train_coco_root = os.path.join(config.data_dir, "train2017")
    if config.dali:
        # Default model, this branch is not be executed, and the alternative branch is dataloaders.dali_pipeline.build_dali_pipeline.
        train_annotate = os.path.join(config.data_dir, "annotations/bbox_only_instances_train2017.json")
        train_trans = SSDTransformer((input_size, input_size), dali=True,
                                     fast_nms=config.fast_nms, fast_cj=config.fast_cj, val=False)
    else:
        train_annotate = os.path.join(config.data_dir, "annotations/instances_train2017.json")
        dboxes = dboxes300_coco()
        train_trans = SSDTransformer((input_size, input_size), dboxes=dboxes, dali=False,
                                     fast_nms=config.fast_nms, fast_cj=config.fast_cj, val=False)

    train_coco = COCODetection(train_coco_root, train_annotate, train_trans)

    if config.distributed:
        train_sampler = GeneralDistributedSampler(train_coco, pad=False)
    else:
        train_sampler = None

    train_loader = DataLoader(train_coco,
                              batch_size=config.train_batch_size,
                              shuffle=(train_sampler is None),
                              sampler=train_sampler,
                              num_workers=config.num_workers,
                              collate_fn=partial(SSDCollator, dali=config.dali))
    return train_loader, len(train_loader), train_sampler


def build_eval_pipe(config):
    # Paths
    input_size = config.input_size
    val_coco_root = os.path.join(config.data_dir, "val2017")
    val_annotate = os.path.join(config.data_dir, "annotations/bbox_only_instances_val2017.json")
    val_trans = SSDTransformer((input_size, input_size), dali=True,
                               fast_nms=config.fast_nms, fast_cj=config.fast_cj, val=True)
    if config.use_coco_ext:
        cocoGt = COCO(annotation_file=val_annotate, use_ext=True)
    else:
        cocoGt = COCO(annotation_file=val_annotate)
    val_coco = COCODetection(val_coco_root, val_annotate, val_trans, cocoGt.dataset)

    if config.distributed:
        val_sampler = GeneralDistributedSampler(val_coco, pad=False)
    else:
        val_sampler = None

    val_dataloader = DataLoader(val_coco,
                                  batch_size=config.eval_batch_size,
                                  shuffle=False, # Note: distributed sampler is shuffled :(
                                  sampler=val_sampler,
                                  num_workers=config.num_workers)

    inv_map = {v:k for k,v in val_coco.label_map.items()}

    return val_dataloader, inv_map, cocoGt


def build_native_pipeline(config, training=True, pipe=None):
    if training:
        return build_train_pipe(config)
    else:
        return build_eval_pipe(config)
