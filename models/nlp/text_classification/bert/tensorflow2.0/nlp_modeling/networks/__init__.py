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

"""Networks are combinations of `tf.keras` layers (and possibly other networks).

They are `tf.keras` models that would not be trained alone. It encapsulates
common network structures like a transformer encoder into an easily
handled object with a standardized configuration.
"""
from nlp_modeling.networks.albert_encoder import AlbertEncoder
from nlp_modeling.networks.bert_encoder import BertEncoder
from nlp_modeling.networks.classification import Classification
from nlp_modeling.networks.encoder_scaffold import EncoderScaffold
from nlp_modeling.networks.mobile_bert_encoder import MobileBERTEncoder
from nlp_modeling.networks.packed_sequence_embedding import PackedSequenceEmbedding
from nlp_modeling.networks.span_labeling import SpanLabeling
from nlp_modeling.networks.span_labeling import XLNetSpanLabeling
from nlp_modeling.networks.xlnet_base import XLNetBase
# Backward compatibility. The modules are deprecated.
TransformerEncoder = BertEncoder
