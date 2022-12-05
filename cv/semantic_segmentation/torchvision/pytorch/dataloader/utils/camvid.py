# Copyright (c) 2022, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# Copyright Declaration: This software, including all of its code and documentation,
# except for the third-party software it contains, is a copyrighted work of Shanghai Iluvatar CoreX
# Semiconductor Co., Ltd. and its affiliates ("Iluvatar CoreX") in accordance with the PRC Copyright
# Law and relevant international treaties, and all rights contained therein are enjoyed by Iluvatar
# CoreX. No user of this software shall have any right, ownership or interest in this software and
# any use of this software shall be in compliance with the terms and conditions of the End User
# License Agreement.


from collections import OrderedDict
import cv2
from functools import lru_cache
import numpy as np
import os
import os.path as osp
from PIL import Image

from .pascal_voc import BaseDataset


class SemanticSeg(BaseDataset):

    def __init__(self,
                 data_dir,
                 anno_dir,
                 data_suffix='.png',
                 anno_suffix='.png',
                 split=None,
                 classes=None,
                 label_rgb=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.data_dir = data_dir
        self.anno_dir = anno_dir
        self.data_suffix = data_suffix
        self.anno_suffix = anno_suffix
        self.label_rgb = label_rgb
        if classes is not None:
            self.CLASSES = classes
            self.COLORS = np.random.randint(0, 255, size=(len(self.CLASSES), 3))
        self.annotations = SemanticSeg.load_annotations(data_dir,
                                                        anno_dir,
                                                        data_suffix,
                                                        anno_suffix,
                                                        split)
        self.image_ids = list(self.annotations.keys())

    @staticmethod
    def load_annotations(data_dir, anno_dir, data_suffix, anno_suffix, split):
        """Load annotation from directory.

        Returns
        -------
        dict[dict]
            All image info of dataset.
        """

        img_infos = dict()
        image_id = 0
        # 如果提供了 split file, 则根据 split file 中的文件名得到数据集的图片和标注
        if split is not None:
            with open(split) as f:
                for line in f:
                    img_name = line.strip()
                    img_file = osp.join(data_dir, img_name + data_suffix)
                    img_info = dict(file_path=img_file)
                    if anno_dir is not None:
                        anno_file = osp.join(anno_dir, img_name + anno_suffix)
                        img_info['anno_path'] = anno_file
                    img_infos[image_id] = img_info
                    image_id += 1
        else:
            for img in scandir(data_dir, data_suffix):
                img_file = osp.join(data_dir, img)
                img_info = dict(file_path=img_file)
                if anno_dir is not None:
                    anno_file = osp.join(anno_dir,
                                         img.replace(data_suffix, anno_suffix))
                    img_info['anno_path'] = anno_file
                img_infos[image_id] = img_info
                image_id += 1

        return img_infos

    def get_data(self, idx):
        image_id = self.image_ids[idx]
        img = self._read_image(image_id)
        mask = self._read_mask(image_id)
        return img, mask

    def get_img_info(self, image_id):
        return self.annotations[image_id]

    @lru_cache(maxsize=None)
    def _read_image(self, image_id):
        img_info = self.get_img_info(image_id)
        image = Image.open(img_info['file_path'])
        return image

    @lru_cache(maxsize=None)
    def _read_mask(self, image_id):
        img_info = self.get_img_info(image_id)
        anno_path = img_info['anno_path']
        # TODO: 当 mask 是 RGB 格式时， 通过 self.COLORS 中 class_id 和 RGB 值
        # 的映射关系， 将 mask 转换为灰度图， 其中每个像素值都对应一个 class_id
        if self.label_rgb:
            # mask = cv2.imread(anno_path)
            # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
            raise NotImplementedError
        else:
            mask = Image.open(anno_path)
        return mask

    @lru_cache(maxsize=None)
    def get_class_name(self, class_id):
        class_dict = {class_name: i for i, class_name in enumerate(self.CLASSES)}
        class_id_dict = {cls_id: name for name, cls_id in class_dict.items()}
        return class_id_dict[class_id]

    @lru_cache(maxsize=None)
    def get_class_color(self, class_id):
        if not isinstance(class_id, str):
            class_name = self.get_class_name(class_id)
        else:
            class_name = class_id
        CLASS_COLOR = OrderedDict(zip(self.CLASSES, self.COLORS))
        return CLASS_COLOR[class_name]


def scandir(dir_path, suffix=None):
    file_paths = []
    for parent, _, fns in sorted(os.walk(dir_path)):
        # 将 dir_path 从 parent 中去除， 使 parent 为相对路径
        parent = parent[len(dir_path):]
        for fn in sorted(fns):
            if fn.endswith(suffix):
                # file 在 dir_path 中的相对路径
                path = os.path.join(parent, fn)
                file_paths.append(path)
    return file_paths



class CamVid(SemanticSeg):
    """CamVid dataset with 32 classes.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.CLASSES = ('Animal', 'Archway', 'Bicyclist', 'Bridge',
                        'Building', 'Car', 'CartLuggagePram', 'Child',
                        'Column_Pole', 'Fence', 'LaneMkgsDriv', 'LaneMkgsNonDriv',
                        'Misc_Text', 'MotorcycleScooter', 'OtherMoving', 'ParkingBlock',
                        'Pedestrian', 'Road', 'RoadShoulder', 'Sidewalk',
                        'SignSymbol', 'Sky', 'SUVPickupTruck', 'TrafficCone',
                        'TrafficLight', 'Train', 'Tree', 'Truck_Bus',
                        'Tunnel', 'VegetationMisc', 'Void', 'Wall')

        self.COLORS = ([64,128,64], [192,0,128], [0,128,192], [0,128,64],
                        [128,0,0], [64,0,128], [64,0,192], [192,128,64],
                        [192,192,128], [64,64,128], [128,0,192], [192,0,64],
                        [128,128,64], [192,0,192], [128,64,64], [64,192,128],
                        [64,64,0], [128,64,128], [128,128,192], [0,0,192],
                        [192,128,128], [128,128,128], [64,128,192], [0,0,64],
                        [0,64,64], [192,64,128], [128,128,0], [192,128,192],
                        [64,0,64], [192,192,0], [0,0,0], [64,192,0])



class CamVid11(SemanticSeg):
    """CamVid dataset with 11 classes.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 如果将 `Unlabelled` (对应 CamVid 中的 `Void` 类) 也算作一类的话, 则总共有 12 类.
        self.CLASSES = ('Sky', 'Building', 'Pole', 'Road',
                        'Pavement', 'Tree', 'SignSymbol', 'Fence',
                        'Car', 'Pedestrian', 'Bicyclist', 'Unlabelled')

        self.COLORS = ([128,128,128], [128,0,0], [192,192,128], [128,64,128],
                       [60,40,222], [128,128,0], [192,128,128], [64,64,128],
                       [64,0,128], [64,64,0], [0,128,192], [0,0,0])


def get_camvid(root, image_set, transforms):
    data_dir = os.path.join(root, image_set)
    anno_dir = os.path.join(root, image_set + "annot")
    return CamVid11(data_dir=data_dir, anno_dir=anno_dir, transform=transforms)

