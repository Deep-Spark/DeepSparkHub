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



import torchvision

from .utils.coco_seg_utils import get_coco
from .utils import presets_segmentation as presets
from .utils.camvid import get_camvid

"""
Examples:

>>> dataset_train, num_classes = get_dataset("/path/to/CamVid11", "camvid", "train")
>>> dataset_val, _ = get_dataset("/path/to/CamVid11", "camvid", "val")

"""


def get_transform(train, base_size, crop_size):
    return presets.SegmentationPresetTrain(base_size, crop_size) if train else presets.SegmentationPresetEval(crop_size)


def get_dataset(dir_path, name, image_set, base_size=540, crop_size=512):
    transform = get_transform(image_set == 'train', base_size, crop_size)
    # name = 'camvid'
    def sbd(*args, **kwargs):
        return torchvision.datasets.SBDataset(*args, mode='segmentation', **kwargs)
    paths = {
        "voc": (dir_path, torchvision.datasets.VOCSegmentation, 21),
        "voc_aug": (dir_path, sbd, 21),
        "coco": (dir_path, get_coco, 21),
        "camvid": (dir_path, get_camvid, 12)
    }
    p, ds_fn, num_classes = paths[name]

    ds = ds_fn(p, image_set=image_set, transforms=transform)
    return ds, num_classes