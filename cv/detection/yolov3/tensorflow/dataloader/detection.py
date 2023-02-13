# Copyright (c) 2022 Iluvatar CoreX. All rights reserved.
# Copyright Declaration: This software, including all of its code and documentation,
# except for the third-party software it contains, is a copyrighted work of Shanghai Iluvatar CoreX
# Semiconductor Co., Ltd. and its affiliates ("Iluvatar CoreX") in accordance with the PRC Copyright
# Law and relevant international treaties, and all rights contained therein are enjoyed by Iluvatar
# CoreX. No user of this software shall have any right, ownership or interest in this software and
# any use of this software shall be in compliance with the terms and conditions of the End User
# License Agreement.


from .utils.coco_utils import get_coco, get_coco_kp
from .utils import presets_detection as presets
from .utils.pascal_voc import get_voc

"""
Examples:

>>> dataset_train, num_classes = get_dataset("voc", "train", "/path/to/VOC2012_sample")
>>> dataset_val, _ = get_dataset("voc", "val", "/path/to/VOC2012_sample")
"""


def get_transform(train, data_augmentation="ssd"):
    return presets.DetectionPresetTrain(data_augmentation) if train else presets.DetectionPresetEval()


def get_dataset(name, image_set, data_path):
    transform = get_transform(image_set.lower() == "train")
    paths = {
        "coco": (data_path, get_coco, 91),
        "coco_kp": (data_path, get_coco_kp, 2),
        "voc": (data_path, get_voc, 21)
    }
    p, ds_fn, num_classes = paths[name]

    ds = ds_fn(p, image_set=image_set, transforms=transform)
    return ds, num_classes