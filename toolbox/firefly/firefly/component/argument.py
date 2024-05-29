# Copyright (c) 2024, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
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


from dataclasses import dataclass, field
from typing import Optional, Union
import enum


class PromptEncoderReparameterizationType(str, enum.Enum):
    MLP = "MLP"
    LSTM = "LSTM"


@dataclass
class CustomizedArguments:
    """
    一些自定义参数
    """
    max_seq_length: int = field(metadata={"help": "输入最大长度"})
    train_file: str = field(metadata={"help": "训练集。如果task_type=pretrain，请指定文件夹，将扫描其下面的所有jsonl文件"})
    model_name_or_path: str = field(metadata={"help": "预训练权重路径"})
    eval_file: Optional[str] = field(default="", metadata={"help": "验证集"})
    tokenize_num_workers: int = field(default=1, metadata={"help": ""})
    task_type: str = field(default="sft", metadata={"help": "预训练任务：[sft, pretrain, dpo]"})
    template_name: str = field(default="", metadata={"help": "sft时的数据格式"})
    peft_type: str = field(default="sft", metadata={"help": "微调方法：[lora, qlora, ptuning, ptuning_v2]"})
    


@dataclass
class QLoRAArguments(CustomizedArguments):
    """
    一些自定义参数
    """
    lora_rank: Optional[int] = field(default=64, metadata={"help": "lora rank"})
    lora_alpha: Optional[int] = field(default=16, metadata={"help": "lora alpha"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "lora dropout"})
    
@dataclass
class PtuningArguments(CustomizedArguments):
    """
    一些自定义参数
    """
    num_virtual_tokens: int = field(default=None, metadata={"help": "虚拟tokens长度"})
    num_attention_heads: Optional[int] = field(default=None, metadata={"help": "Number of attention heads"})
    encoder_num_layers: int = field(default=2, metadata={"help": "The number of layers of the prompt encoder"})
    encoder_reparameterization_type: Union[str, PromptEncoderReparameterizationType] = field(
        default=PromptEncoderReparameterizationType.MLP, metadata={"help": "How to reparameterize the prompt encoder"})
    encoder_hidden_size: int = field(default=None, metadata={"help": "The hidden size of the prompt encoder"})
    token_dim: int = field(default=None, metadata={"help": "The hidden size of the prompt encoder"})
    num_layers: int = field(default=None, metadata={"help": "The hidden size of the prompt encoder"})
    
@dataclass
class PrefixArguments(CustomizedArguments):
    """
    一些自定义参数
    """
    num_virtual_tokens: int = field(default=None, metadata={"help": "虚拟tokens长度"})
    token_dim: int = field(default=None, metadata={"help": "The hidden size of the prompt encoder"})
    num_attention_heads: Optional[int] = field(default=None, metadata={"help": "Number of attention heads"})
    num_layers: int = field(default=None, metadata={"help": "The hidden size of the prompt encoder"})
    encoder_hidden_size: int = field(default=None, metadata={"help": "The hidden size of the prompt encoder"})
    prefix_projection: bool = field(default=False, metadata={"help": "The hidden size of the prompt encoder"})
    num_transformer_submodules: Optional[int] = field(
        default=None, metadata={"help": "Number of transformer submodules"}
    )
    
    

@dataclass
class DPOArguments:
    """
    一些自定义参数
    """
    max_seq_length: int = field(metadata={"help": "输入最大长度"})
    max_prompt_length: Optional[int] = field(metadata={"help": "max length of prompt"})

    train_file: str = field(metadata={"help": "训练集"})
    model_name_or_path: str = field(metadata={"help": "预训练权重路径"})
    eval_file: Optional[str] = field(default="", metadata={"help": "the file of training data"})

    # 定义template，单轮对话prompt的拼接格式为：{system}{conv_begin}{human_begin}你好{human_end}{assistant_begin}
    system: int = field(default='', metadata={"help": ""})
    conv_begin: int = field(default='', metadata={"help": ""})
    human_begin: int = field(default='', metadata={"help": ""})
    human_end: int = field(default='', metadata={"help": ""})
    assistant_begin: int = field(default='', metadata={"help": ""})
    assistant_end: int = field(default='', metadata={"help": ""})

    use_lora: bool = field(default=False, metadata={"help": "预训练任务：[sft, pretrain]"})
    beta: Optional[float] = field(default=0.1, metadata={"help": "the beta parameter for DPO loss"})
    lora_rank: Optional[int] = field(default=64, metadata={"help": "lora rank"})
    lora_alpha: Optional[int] = field(default=16, metadata={"help": "lora alpha"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "lora dropout"})


@dataclass
class LOMOArguments:
    """
    LOMO训练的自定义参数
    """
    max_seq_length: int = field(metadata={"help": "输入最大长度"})
    train_file: str = field(metadata={"help": "训练集"})
    model_name_or_path: str = field(metadata={"help": "预训练权重路径"})
    clip_grad_norm: float = field(metadata={"help": "Maximum gradient normalized value (for gradient clipping)."})
    clip_grad_value: float = field(default=None, metadata={"help": "Maximum gradient value (for gradient clipping)."})
    eval_file: Optional[str] = field(default="", metadata={"help": "the file of training data"})
