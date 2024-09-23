# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# All Rights Reserved.

from .fused_layer_norm import MixedFusedLayerNorm as LayerNorm
from .fused_layer_norm import MixedFusedRMSNormResidual as RMSNormResidual
from .rms_norm import RMSNorm

from .distributed import DistributedDataParallel
#from .bert_model import BertModel
from .gpt_model import GPTModel, GPTModelPipe
from .t5_model import T5Model
from .language_model import get_language_model
from .module import Float16Module
