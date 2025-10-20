# Copyright (c) OpenMMLab. All rights reserved.
from ocrdet.datasets_.builder import DATASETS, build_dataloader, build_dataset

from . import utils
from .base_dataset import BaseDataset
from .pipelines import CustomFormatBundle
from .ocr_dataset import OCRDataset
from .uniform_concat_dataset import UniformConcatDataset
from .utils import *  # NOQA

__all__ = [
    'DATASETS', 'build_dataloader', 'build_dataset', "OCRDataset"
    'BaseDataset', 'CustomFormatBundle', 'UniformConcatDataset'
]

__all__ += utils.__all__