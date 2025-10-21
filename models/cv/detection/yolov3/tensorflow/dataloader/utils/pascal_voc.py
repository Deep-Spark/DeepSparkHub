# Copyright (c) 2022 Iluvatar CoreX. All rights reserved.
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.


from collections import OrderedDict
import cv2
# from functools import lru_cache
import os
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image



from abc import ABCMeta, abstractmethod

try:
    import torch
    from torch.utils.data import Dataset
except:
    torch = None
    Dataset = object

# 继承 BaseDataset 的 class 需要实现 get_data, 并填充 image_ids 字段
class BaseDataset(Dataset, metaclass=ABCMeta):
    def __init__(self, transform, **kwargs):
        super(BaseDataset, self).__init__()
        self.pipeline = transform
        self.image_ids = []

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image, target = self.get_data(idx)
        if self.pipeline is not None:
            image, target = self.pipeline(image, target)
        return image, target

    @abstractmethod
    def get_data(self, idx):
        pass


class PascalVOC(BaseDataset):
    ALL_CLASSES = ('__background__',
                   'aeroplane', 'bicycle', 'bird', 'boat',
                   'bottle', 'bus', 'car', 'cat', 'chair',
                   'cow', 'diningtable', 'dog', 'horse',
                   'motorbike', 'person', 'pottedplant',
                   'sheep', 'sofa', 'train', 'tvmonitor')

    COLORS = ([0, 0, 0],
              [128, 0, 0], [0, 128, 0], [128, 128, 0],
              [0, 0, 128], [128, 0, 128], [0, 128, 128],
              [128, 128, 128], [64, 0, 0], [192, 0, 0],
              [64, 128, 0], [192, 128, 0], [64, 0, 128],
              [192, 0, 128], [64, 128, 128], [192, 128, 128],
              [0, 64, 0], [128, 64, 0], [0, 192, 0],
              [128, 192, 0], [0, 64, 128])

    CLASS_COLOR = OrderedDict(zip(ALL_CLASSES, COLORS))

    def __init__(self,
                 data_dir,
                 split,
                 image_sets_dir='Main',
                 classes=None,
                 keep_difficult=False,
                 to_torch_tensor=True,
                 **kwargs):
        """Dataset for VOC data.

        Parameters
        ----------
        data_dir : str
            the root of the VOC2007 or VOC2012 dataset, the directory contains the
            following sub-directories:
            Annotations, ImageSets, JPEGImages, SegmentationClass, SegmentationObject.
        split : str
            "train" or "val"
        """
        super().__init__(**kwargs)
        self.data_dir = data_dir
        self.split = split
        image_sets_file = os.path.join(
            self.data_dir, 'ImageSets', image_sets_dir, '%s.txt' % self.split)
        if classes is not None:
            self.CLASSES = classes
            if '__background__' not in classes:
                self.CLASSES.insert(0, '__background__')
        else:
            self.CLASSES = self.ALL_CLASSES

        self.image_ids = PascalVOC.get_image_ids(image_sets_file)
        self.keep_difficult = keep_difficult
        self.class_dict = {class_name: i for i, class_name in enumerate(self.CLASSES)}
        self.class_id_dict = {cls_id: name for name, cls_id in self.class_dict.items()}
        self.to_torch_tensor = to_torch_tensor and torch is not None

    def get_data(self, idx):
        image_id = self.image_ids[idx]
        bboxes, labels, difficult = self._get_annotation(image_id)
        # 忽略掉 difficult 的 object
        if not self.keep_difficult:
            bboxes = bboxes[difficult==0]
            labels = labels[difficult==0]
        image = self._read_image(image_id)
        img_info = self.get_img_info(idx)
        # 因为使用了 lru cache， 为了确保 cache 中的数据不被 pipeline 更改，
        # 需要将原始数据的拷贝送入 pipeline
        area = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
        image_id = image_id.replace('-', '')
        image_id = int(image_id)

        if self.to_torch_tensor:
            target = dict(boxes=torch.from_numpy(bboxes),
                        labels=torch.from_numpy(labels),
                        image_id=torch.tensor(image_id),
                        area=torch.from_numpy(area),
                        iscrowd=torch.tensor([False] * len(bboxes)))
        else:
            target = dict(boxes=torch.from_numpy(bboxes.copy()),
                          labels=torch.from_numpy(labels.copy()),
                          image_id=image_id,
                          area=area.copy(),
                          iscrowd=[False] * len(bboxes))
        return image, target

    # 获取索引为 idx 的 sample 的标注信息
    def get_annotation(self, idx):
        image_id = self.image_ids[idx]
        bboxes, labels, difficult = self._get_annotation(image_id)
        anno = dict(image_id=image_id,
                    bboxes=bboxes,
                    labels=labels,
                    difficult=difficult)
        return anno

    # @lru_cache(maxsize=None)
    def _get_annotation(self, image_id):
        annotation_file = os.path.join(self.data_dir, "Annotations", "%s.xml" % image_id)
        objects = ET.parse(annotation_file).findall("object")
        boxes = []
        labels = []
        difficult = []
        for obj in objects:
            class_name = obj.find('name').text.lower().strip()
            if class_name not in self.CLASSES:
                continue
            bbox = obj.find('bndbox')
            # VOC dataset format follows Matlab, in which indices start from 0
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1
            boxes.append([x1, y1, x2, y2])
            labels.append(self.class_dict[class_name])
            difficult_str = obj.find('difficult').text
            difficult.append(int(difficult_str) if difficult_str else 0)

        return (np.array(boxes, dtype=np.float32),
                np.array(labels, dtype=np.int64),
                np.array(difficult, dtype=np.uint8))

    @staticmethod
    def get_image_ids(image_sets_file):
        ids = []
        with open(image_sets_file) as f:
            for line in f:
                ids.append(line.rstrip())
        return ids

    # @lru_cache(maxsize=None)
    def get_img_info(self, idx):
        image_id = self.image_ids[idx]
        annotation_file = os.path.join(self.data_dir, "Annotations", "%s.xml" % image_id)
        anno = ET.parse(annotation_file).getroot()
        size = anno.find("size")
        im_info = tuple(map(int, (size.find("height").text, size.find("width").text)))
        image_path = os.path.join(self.data_dir, "JPEGImages", "%s.jpg" % image_id)
        return dict(id=image_id,
                    file_path=image_path,
                    height=im_info[0],
                    width=im_info[1])

    # @lru_cache(maxsize=None)
    def _read_image(self, image_id):
        image_file = os.path.join(self.data_dir, "JPEGImages", "%s.jpg" % image_id)
        image = Image.open(image_file)
        return image

    def get_class_name(self, class_id):
        return self.class_id_dict[class_id]

    def get_class_color(self, class_id):
        if not isinstance(class_id, str):
            class_name = self.get_class_name(class_id)
        else:
            class_name = class_id
        return self.CLASS_COLOR[class_name]



class VOCInstanceSeg(PascalVOC):
    IGNORE_IMAGE_IDS = ("2008_005953", "2008_007355")
    def __init__(self, **kwargs):
        super().__init__(image_sets_dir='Segmentation', **kwargs)

        if "ignore_image_id" in kwargs and (isinstance(kwargs["ignore_image_id"], list) or isinstance(kwargs["ignore_image_id"], tuple)):
            self.ignore_ids = kwargs["ignore_image_id"]
        else:
            self.ignore_ids = self.IGNORE_IMAGE_IDS

        self.image_ids = [id for id in self.image_ids if id not in self.ignore_ids]

    def get_data(self, idx):
        image_id = self.image_ids[idx]
        bboxes, labels, masks, difficult = self._get_annotation(image_id)
        # 忽略掉 difficult 的 object
        if not self.keep_difficult:
            bboxes = bboxes[difficult==0]
            labels = labels[difficult==0]
            masks = np.transpose(masks, axes=(2, 0, 1))
            masks = masks[difficult==0]
            # masks = np.transpose(masks, axes=(1, 2, 0))
        image = self._read_image(image_id)
        img_info = self.get_img_info(idx)
        # 因为使用了 lru cache， 为了确保 cache 中的数据不被 pipeline 更改，
        # 需要将原始数据的拷贝送入 pipeline
        area = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
        image_id = image_id.replace('-', '')
        image_id = int(image_id)
        if self.to_torch_tensor:
            target = dict(boxes=torch.from_numpy(bboxes),
                          labels=torch.from_numpy(labels),
                          image_id=torch.tensor(image_id),
                          area=torch.from_numpy(area),
                          iscrowd=torch.tensor([False] * len(bboxes)),
                          masks=torch.from_numpy(masks))
        else:
            target = dict(boxes=bboxes.copy(),
                          labels=labels.copy(),
                          image_id=image_id,
                          area=area.copy(),
                          iscrowd=[False] * len(bboxes),
                          masks=masks.copy())
        return image, target

    # @lru_cache(maxsize=None)
    def _get_annotation(self, image_id):
        annotation_file = os.path.join(self.data_dir, "Annotations", "%s.xml" % image_id)
        objects = ET.parse(annotation_file).findall("object")
        mask = self._read_mask(image_id)
        boxes = []
        labels = []
        objmasks = []
        difficult = []
        for obj in objects:
            class_name = obj.find('name').text.lower().strip()
            if class_name not in self.CLASSES:
                continue
            bbox = obj.find('bndbox')
            # VOC dataset format follows Matlab, in which indices start from 0
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1
            boxes.append([x1, y1, x2, y2])
            labels.append(self.class_dict[class_name])
            difficult_str = obj.find('difficult').text
            difficult.append(int(difficult_str) if difficult_str else 0)

            class_color = self.get_class_color(class_name)
            objmask = self._get_objmask(mask, (x1, y1, x2, y2), class_color)
            objmasks.append(objmask)

        if len(objmasks) == 1:
            objmasks = np.expand_dims(objmasks[0], axis=-1)
        else:
            objmasks = np.stack(objmasks, axis=-1)

        # masks 的 shape 为 (height, width, N), 目的是方便复用对 img 的 transform 操作
        return (np.array(boxes, dtype=np.float32),
                np.array(labels, dtype=np.int64),
                objmasks,
                np.array(difficult, dtype=np.uint8))

    def _read_mask(self, image_id):
        mask_file = os.path.join(self.data_dir, "SegmentationClass", "%s.png" % image_id)
        mask = cv2.imread(mask_file)
        return mask

    def _get_objmask(self, mask, bbox, class_color):
        im_heigt, im_width = mask.shape[0], mask.shape[1]
        xmin, ymin, xmax, ymax = [int(coor) for coor in bbox]

        objmask = np.zeros((im_heigt, im_width), dtype=np.uint8)
        for i in range(ymin, ymax):
            for j in range(xmin, xmax):
                # mask 为 BGR， class_color 为 RGB
                if mask[i,j,0] == class_color[2] and \
                   mask[i,j,1] == class_color[1] and \
                   mask[i,j,2] == class_color[0]:
                    objmask[i,j] = 1

        return objmask


def get_voc(root, image_set, transforms, to_torch_tensor=True):
    return VOCInstanceSeg(data_dir=root, split=image_set, transform=transforms, to_torch_tensor=to_torch_tensor)