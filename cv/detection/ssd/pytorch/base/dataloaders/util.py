# MIT License
#
# Copyright (c) 2018 kuangliu
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data
from PIL import Image
from xml.etree import ElementTree
import os
import glob
from pathlib import Path
import numpy as np
import random
import itertools
import torch.nn.functional as F
try:
    import ujson as json
except ImportError:
    import json
import gc
import time
# import bz2
# import pickle
from math import sqrt

from box_coder import calc_iou_tensor, Encoder


class DefaultBoxes(object):
    def __init__(self, fig_size, feat_size, steps, scales, aspect_ratios, \
                 scale_xy=0.1, scale_wh=0.2):

        self.feat_size = feat_size
        self.fig_size = fig_size

        self.scale_xy_ = scale_xy
        self.scale_wh_ = scale_wh

        # According to https://github.com/weiliu89/caffe
        # Calculation method slightly different from paper
        self.steps = steps
        self.scales = scales

        fk = fig_size / np.array(steps)
        self.aspect_ratios = aspect_ratios

        self.default_boxes = []
        # size of feature and number of feature
        for idx, sfeat in enumerate(self.feat_size):

            sk1 = scales[idx] / fig_size
            sk2 = scales[idx + 1] / fig_size
            sk3 = sqrt(sk1 * sk2)
            all_sizes = [(sk1, sk1), (sk3, sk3)]

            for alpha in aspect_ratios[idx]:
                w, h = sk1 * sqrt(alpha), sk1 / sqrt(alpha)
                all_sizes.append((w, h))
                all_sizes.append((h, w))
            for w, h in all_sizes:
                for i, j in itertools.product(range(sfeat), repeat=2):
                    cx, cy = (j + 0.5) / fk[idx], (i + 0.5) / fk[idx]
                    self.default_boxes.append((cx, cy, w, h))

        self.dboxes = torch.tensor(self.default_boxes, dtype=torch.float)
        self.dboxes.clamp_(min=0, max=1)
        # For IoU calculation
        self.dboxes_ltrb = self.dboxes.clone()
        self.dboxes_ltrb[:, 0] = self.dboxes[:, 0] - 0.5 * self.dboxes[:, 2]
        self.dboxes_ltrb[:, 1] = self.dboxes[:, 1] - 0.5 * self.dboxes[:, 3]
        self.dboxes_ltrb[:, 2] = self.dboxes[:, 0] + 0.5 * self.dboxes[:, 2]
        self.dboxes_ltrb[:, 3] = self.dboxes[:, 1] + 0.5 * self.dboxes[:, 3]

    @property
    def scale_xy(self):
        return self.scale_xy_

    @property
    def scale_wh(self):
        return self.scale_wh_

    def __call__(self, order="ltrb"):
        if order == "ltrb": return self.dboxes_ltrb
        if order == "xywh": return self.dboxes


# This function is from https://github.com/chauhan-utk/ssd.DomainAdaptation.
class SSDCropping(object):
    """ Cropping for SSD, according to original paper
        Choose between following 3 conditions:
        1. Preserve the original image
        2. Random crop minimum IoU is among 0.1, 0.3, 0.5, 0.7, 0.9
        3. Random crop
        Reference to https://github.com/chauhan-utk/ssd.DomainAdaptation
    """

    def __init__(self, num_cropping_iterations=1):

        self.sample_options = (
            # Do nothing
            None,
            # min IoU, max IoU
            (0.1, None),
            (0.3, None),
            (0.5, None),
            (0.7, None),
            (0.9, None),
            # no IoU requirements
            (None, None),
        )
        # Implementation uses 1 iteration to find a possible candidate, this
        # was shown to produce the same mAP as using more iterations.
        self.num_cropping_iterations = num_cropping_iterations

    def __call__(self, img, img_size, bboxes, labels):

        # Ensure always return cropped image
        while True:
            mode = random.choice(self.sample_options)

            if mode is None:
                return img, img_size, bboxes, labels

            htot, wtot = img_size

            min_iou, max_iou = mode
            min_iou = float("-inf") if min_iou is None else min_iou
            max_iou = float("+inf") if max_iou is None else max_iou

            for _ in range(self.num_cropping_iterations):
                # suze of each sampled path in [0.1, 1] 0.3*0.3 approx. 0.1
                w = random.uniform(0.3, 1.0)
                h = random.uniform(0.3, 1.0)

                if w / h < 0.5 or w / h > 2:
                    continue

                # left 0 ~ wtot - w, top 0 ~ htot - h
                left = random.uniform(0, 1.0 - w)
                top = random.uniform(0, 1.0 - h)

                right = left + w
                bottom = top + h

                ious = calc_iou_tensor(bboxes, torch.tensor(
                    [[left, top, right, bottom]]))

                # tailor all the bboxes and return
                if not ((ious > min_iou) & (ious < max_iou)).all():
                    continue

                # discard any bboxes whose center not in the cropped image
                xc = 0.5 * (bboxes[:, 0] + bboxes[:, 2])
                yc = 0.5 * (bboxes[:, 1] + bboxes[:, 3])

                masks = (xc > left) & (xc < right) & (yc > top) & (yc < bottom)

                # if no such boxes, continue searching again
                if not masks.any():
                    continue

                bboxes[bboxes[:, 0] < left, 0] = left
                bboxes[bboxes[:, 1] < top, 1] = top
                bboxes[bboxes[:, 2] > right, 2] = right
                bboxes[bboxes[:, 3] > bottom, 3] = bottom

                # print(left, top, right, bottom)
                # print(labels, bboxes, masks)
                bboxes = bboxes[masks, :]
                labels = labels[masks]

                left_idx = int(left * wtot)
                top_idx = int(top * htot)
                right_idx = int(right * wtot)
                bottom_idx = int(bottom * htot)
                # print(left_idx,top_idx,right_idx,bottom_idx)
                # img = img[:, top_idx:bottom_idx, left_idx:right_idx]
                img = img.crop((left_idx, top_idx, right_idx, bottom_idx))

                bboxes[:, 0] = (bboxes[:, 0] - left) / w
                bboxes[:, 1] = (bboxes[:, 1] - top) / h
                bboxes[:, 2] = (bboxes[:, 2] - left) / w
                bboxes[:, 3] = (bboxes[:, 3] - top) / h

                htot = bottom_idx - top_idx
                wtot = right_idx - left_idx
                return img, (htot, wtot), bboxes, labels


class ToTensor(object):
    def __init__(self):
        pass

    def __call__(self, img):
        img = torch.Tensor(np.array(img))
        # Transform from HWC to CHW
        img = img.permute(2, 0, 1).div(255)
        return img


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, bboxes):
        if random.random() < self.p:
            bboxes[:, 0], bboxes[:, 2] = 1.0 - bboxes[:, 2], 1.0 - bboxes[:, 0]
            return image.transpose(Image.FLIP_LEFT_RIGHT), bboxes
        return image, bboxes


# Do data augumentation
class SSDTransformer(object):
    """ SSD Data Augumentation, according to original paper
        Composed by several steps:
        Cropping
        Resize
        Flipping
        Jittering
    """
    def __init__(self, size=(300, 300), dboxes=None, dali=False, fast_nms=False, fast_cj=False, val=False, num_cropping_iterations=1):

        # define vgg16 mean
        self.size = size
        self.val = val
        self.dali = dali
        self.crop = SSDCropping(num_cropping_iterations=num_cropping_iterations)
        if self.dali:
            self.dboxes_ = None
            self.encoder = None
        else:
            self.dboxes_ = dboxes  # DefaultBoxes300()
            self.encoder = Encoder(self.dboxes_, fast_nms)

        if fast_cj:
            from fused_color_jitter import FusedColorJitter
            self.img_trans = transforms.Compose([
                transforms.Resize(self.size),
                FusedColorJitter(),
                ToTensor(),
            ])
        else:
            self.img_trans = transforms.Compose([
                transforms.Resize(self.size),
                transforms.ColorJitter(brightness=0.125, contrast=0.5,
                                       saturation=0.5, hue=0.05
                                       ),
                transforms.ToTensor()
            ])
        self.hflip = RandomHorizontalFlip()

        # All Pytorch Tensor will be normalized
        # https://discuss.pytorch.org/t/how-to-preprocess-input-for-pre-trained-networks/683
        normalization_mean = [0.485, 0.456, 0.406]
        normalization_std = [0.229, 0.224, 0.225]
        self.normalize = transforms.Normalize(mean=normalization_mean,
                                              std=normalization_std)

        self.trans_val = transforms.Compose([
            transforms.Resize(self.size),
            transforms.ToTensor(),
            self.normalize])

    @property
    def dboxes(self):
        return self.dboxes_

    def __call__(self, img, img_size, bbox=None, label=None, max_num=200):
        # img = torch.tensor(img)
        if self.val:
            bbox_out = torch.zeros(max_num, 4)
            label_out = torch.zeros(max_num, dtype=torch.long)
            bbox_out[:bbox.size(0), :] = bbox
            label_out[:label.size(0)] = label
            return self.trans_val(img), img_size, bbox_out, label_out

        # random crop
        img, img_size, bbox, label = self.crop(img, img_size, bbox, label)

        # random horiz. flip
        img, bbox = self.hflip(img, bbox)

        # [Resize, ColorJitter, ToTensor]
        img = self.img_trans(img).contiguous()

        img = self.normalize(img)

        if not self.dali:
            bbox, label = self.encoder.encode(bbox, label)

        return img, img_size, bbox, label


# Implement a datareader for COCO dataset
class COCODetection(data.Dataset):
    def __init__(self, img_folder, annotate_file, transform=None, data=None):
        self.img_folder = img_folder
        self.annotate_file = annotate_file

        if data:
            self.data = data
        else:
            # Start processing annotation
            with open(annotate_file) as fin:
                # loading huge json files tends to cause the gc (cycle collector) to
                # waste a lot of time so:
                gc_old = gc.isenabled()
                gc.disable()

                self.data = json.load(fin)

                if gc_old:
                    gc.enable()

        self.images = {}

        self.label_map = {}
        self.label_info = {}
        # print("Parsing COCO data...")
        start_time = time.time()
        # 0 stand for the background
        cnt = 0
        self.label_info[cnt] = "background"
        for cat in self.data["categories"]:
            cnt += 1
            self.label_map[cat["id"]] = cnt
            self.label_info[cnt] = cat["name"]

        # build inference for images
        for img in self.data["images"]:
            img_id = img["id"]
            img_name = img["file_name"]
            img_size = (img["height"], img["width"])
            # print(img_name)
            if img_id in self.images: raise Exception("dulpicated image record")
            self.images[img_id] = (img_name, img_size, [])

        # read bboxes
        for bboxes in self.data["annotations"]:
            img_id = bboxes["image_id"]
            category_id = bboxes["category_id"]
            bbox = bboxes["bbox"]
            bbox_label = self.label_map[bboxes["category_id"]]
            self.images[img_id][2].append((bbox, bbox_label))

        for k, v in list(self.images.items()):
            if len(v[2]) == 0:
                # print("empty image: {}".format(k))
                self.images.pop(k)

        self.img_keys = list(self.images.keys())
        self.transform = transform
        # print("End parsing COCO data, total time {}".format(time.time()-start_time))

    @property
    def labelnum(self):
        return len(self.label_info)

    # @staticmethod
    # def load(pklfile):
    #     # print("Loading from {}".format(pklfile))
    #     with bz2.open(pklfile, "rb") as fin:
    #         ret = pickle.load(fin)
    #     return ret

    # def save(self, pklfile):
    #     # print("Saving to {}".format(pklfile))
    #     with bz2.open(pklfile, "wb") as fout:
    #         pickle.dump(self, fout)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_id = self.img_keys[idx]
        img_data = self.images[img_id]
        fn = img_data[0]
        img_path = os.path.join(self.img_folder, fn)
        s = time.time()
        img = Image.open(img_path).convert("RGB")
        e = time.time()
        decode_time = e - s

        htot, wtot = img_data[1]
        bbox_sizes = []
        bbox_labels = []

        # for (xc, yc, w, h), bbox_label in img_data[2]:
        for (l, t, w, h), bbox_label in img_data[2]:
            r = l + w
            b = t + h
            #l, t, r, b = xc - 0.5*w, yc - 0.5*h, xc + 0.5*w, yc + 0.5*h
            bbox_size = (l/wtot, t/htot, r/wtot, b/htot)
            # filter out zero-size bboxes
            if l == r or t == b:
                continue
            bbox_sizes.append(bbox_size)
            bbox_labels.append(bbox_label)

        bbox_sizes = torch.tensor(bbox_sizes)
        bbox_labels =  torch.tensor(bbox_labels)

        s = time.time()
        if self.transform != None:
            img, (htot, wtot), bbox_sizes, bbox_labels = \
                self.transform(img, (htot, wtot), bbox_sizes, bbox_labels)
        else:
            pass # img = transforms.ToTensor()(img)

        return img, img_id, (htot, wtot), bbox_sizes, bbox_labels