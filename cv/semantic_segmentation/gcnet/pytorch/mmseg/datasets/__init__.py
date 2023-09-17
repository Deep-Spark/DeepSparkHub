# Copyright (c) OpenMMLab. All rights reserved.
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
# from .ade import ADE20KDataset
from .builder import DATASETS, PIPELINES, build_dataloader, build_dataset
#from .chase_db1 import ChaseDB1Dataset
from .cityscapes import CityscapesDataset
# from .coco_stuff import COCOStuffDataset
from .custom import CustomDataset
# from .dark_zurich import DarkZurichDataset
from .dataset_wrappers import ConcatDataset, RepeatDataset
# from .drive import DRIVEDataset
# from .hrf import HRFDataset
# from .loveda import LoveDADataset
# from .night_driving import NightDrivingDataset
# from .pascal_context import PascalContextDataset, PascalContextDataset59
# # from .stare import STAREDataset
# from .voc import PascalVOCDataset

