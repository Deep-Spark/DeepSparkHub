# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import importlib
from typing import List, Optional, Type

import torch.nn as nn

# Supported models using HF Rmpad
# TODO(sgm): HF may supported more than listed here, we should add more after testing
from transformers import LlamaConfig, MistralConfig, GemmaConfig, Qwen2Config

_REOVEPAD_MODELS = {'llama': LlamaConfig, 'mistral': MistralConfig, 'gemma': GemmaConfig, 'qwen2': Qwen2Config}


def check_model_support_rmpad(model_type: str):
    assert isinstance(model_type, str)
    if not model_type in _REOVEPAD_MODELS.keys():
        raise ValueError(f"Model architecture {model_type} is not supported for now. "
                         f"RMPad supported architectures: {_REOVEPAD_MODELS.keys()}."
                         f"Please set `use_remove_padding=False` in the model config.")


# Supported models in Megatron-LM
# Architecture -> (module, class).
_MODELS = {
    "LlamaForCausalLM":
        ("llama", ("ParallelLlamaForCausalLMRmPadPP", "ParallelLlamaForValueRmPadPP", "ParallelLlamaForCausalLMRmPad")),
    "Qwen2ForCausalLM":
        ("qwen2", ("ParallelQwen2ForCausalLMRmPadPP", "ParallelQwen2ForValueRmPadPP", "ParallelQwen2ForCausalLMRmPad")),
    "MistralForCausalLM": ("mistral", ("ParallelMistralForCausalLMRmPadPP", "ParallelMistralForValueRmPadPP",
                                       "ParallelMistralForCausalLMRmPad"))
}


# return model class
class ModelRegistry:

    @staticmethod
    def load_model_cls(model_arch: str, value=False) -> Optional[Type[nn.Module]]:
        if model_arch not in _MODELS:
            return None

        megatron = "megatron"

        module_name, model_cls_name = _MODELS[model_arch]
        if not value:  # actor/ref
            model_cls_name = model_cls_name[0]
        elif value:  # critic/rm
            model_cls_name = model_cls_name[1]

        module = importlib.import_module(f"verl.models.{module_name}.{megatron}.modeling_{module_name}_megatron")
        return getattr(module, model_cls_name, None)

    @staticmethod
    def get_supported_archs() -> List[str]:
        return list(_MODELS.keys())
