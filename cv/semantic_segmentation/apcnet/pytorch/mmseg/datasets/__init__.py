# Copyright (c) OpenMMLab. All rights reserved.
from .ade import ADE20KDataset
from .builder import DATASETS, PIPELINES, build_dataloader, build_dataset
from .cityscapes import CityscapesDataset
from .coco_stuff import COCOStuffDataset
from .custom import CustomDataset
from .dataset_wrappers import (ConcatDataset, MultiImageMixDataset,
                               RepeatDataset)
from .pascal_context import PascalContextDataset, PascalContextDataset59
from .voc import PascalVOCDataset
