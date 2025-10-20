# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Layers are the fundamental building blocks for NLP models.

They can be used to assemble new `tf.keras` layers or models.
"""
# pylint: disable=wildcard-import
from nlp_modeling.layers.attention import *
from nlp_modeling.layers.bigbird_attention import BigBirdAttention
from nlp_modeling.layers.bigbird_attention import BigBirdMasks
from nlp_modeling.layers.cls_head import *
from nlp_modeling.layers.dense_einsum import DenseEinsum
from nlp_modeling.layers.gated_feedforward import GatedFeedforward
from nlp_modeling.layers.gaussian_process import RandomFeatureGaussianProcess
from nlp_modeling.layers.kernel_attention import KernelAttention
from nlp_modeling.layers.kernel_attention import KernelMask
from nlp_modeling.layers.masked_lm import MaskedLM
from nlp_modeling.layers.masked_softmax import MaskedSoftmax
from nlp_modeling.layers.mat_mul_with_margin import MatMulWithMargin
from nlp_modeling.layers.mobile_bert_layers import MobileBertEmbedding
from nlp_modeling.layers.mobile_bert_layers import MobileBertMaskedLM
from nlp_modeling.layers.mobile_bert_layers import MobileBertTransformer
from nlp_modeling.layers.multi_channel_attention import *
from nlp_modeling.layers.on_device_embedding import OnDeviceEmbedding
from nlp_modeling.layers.position_embedding import RelativePositionBias
from nlp_modeling.layers.position_embedding import RelativePositionEmbedding
from nlp_modeling.layers.relative_attention import MultiHeadRelativeAttention
from nlp_modeling.layers.relative_attention import TwoStreamRelativeAttention
from nlp_modeling.layers.rezero_transformer import ReZeroTransformer
from nlp_modeling.layers.self_attention_mask import SelfAttentionMask
from nlp_modeling.layers.spectral_normalization import *
from nlp_modeling.layers.talking_heads_attention import TalkingHeadsAttention
from nlp_modeling.layers.text_layers import BertPackInputs
from nlp_modeling.layers.text_layers import BertTokenizer
from nlp_modeling.layers.text_layers import SentencepieceTokenizer
from nlp_modeling.layers.tn_transformer_expand_condense import TNTransformerExpandCondense
from nlp_modeling.layers.transformer import *
from nlp_modeling.layers.transformer_scaffold import TransformerScaffold
from nlp_modeling.layers.transformer_xl import TransformerXL
from nlp_modeling.layers.transformer_xl import TransformerXLBlock
