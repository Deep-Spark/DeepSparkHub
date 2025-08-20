# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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
"""
Utilities for using tensor_parallel in megatron
"""
from typing import Dict
import torch
from torch.nn import init
import torch.distributed as dist
from megatron.core import ModelParallelConfig
from megatron.core import parallel_state as mpu, tensor_parallel
import verl.utils.torch_functional as verl_F


def update_kwargs_with_config(dictionary: Dict, config: ModelParallelConfig):
    dictionary['config'] = config
    return dictionary


def get_default_kwargs_for_model_parallel_config():
    model_parallel_config_kwargs = {
        'params_dtype': torch.float32,
        'use_cpu_initialization': False,
        'perform_initialization': True,
        'gradient_accumulation_fusion': False,
        'sequence_parallel': False,
    }
    return model_parallel_config_kwargs


def get_default_model_parallel_config():
    return ModelParallelConfig(**get_default_kwargs_for_model_parallel_config())


def get_common_default_kwargs_for_parallel_linear():
    default_model_parallel_config = get_default_model_parallel_config()
    common_default_kwargs = {
        'init_method': init.xavier_normal_,
        'stride': 1,
        'keep_master_weight_for_test': False,
        'config': default_model_parallel_config,
    }
    return common_default_kwargs


def get_default_kwargs_for_column_parallel_linear():
    model_parallel_config_kwargs = get_default_kwargs_for_model_parallel_config()
    column_parallel_config_kwargs = {
        'async_tensor_model_parallel_allreduce': False,
    }
    model_parallel_config_kwargs.update(column_parallel_config_kwargs)
    column_default_kwargs = {
        'config': ModelParallelConfig(**model_parallel_config_kwargs),
    }
    common_default_kwargs = get_common_default_kwargs_for_parallel_linear()
    common_default_kwargs.update(column_default_kwargs)
    return common_default_kwargs


def get_default_kwargs_for_row_parallel_linear():
    common_default_kwargs = get_common_default_kwargs_for_parallel_linear()
    return common_default_kwargs


def get_default_kwargs_for_parallel_embedding():
    model_parallel_config_kwargs = get_default_kwargs_for_model_parallel_config()
    embedding_default_kwargs = {
        'init_method': init.xavier_normal_,
        'config': ModelParallelConfig(**model_parallel_config_kwargs),
    }
    return embedding_default_kwargs


def is_tensor_parallel_param(param):
    return (hasattr(param, 'tensor_model_parallel') and param.tensor_model_parallel)


def get_tensor_parallel_partition_dim(param):
    assert is_tensor_parallel_param(param)
    return param.partition_dim


def get_tensor_parallel_partition_stride(param):
    assert is_tensor_parallel_param(param)
    return param.partition_stride


class _VocabParallelEntropy(torch.autograd.Function):

    @staticmethod
    def forward(ctx, vocab_parallel_logits: torch.Tensor) -> torch.Tensor:
        logits_max = vocab_parallel_logits.max(dim=-1, keepdim=True).values
        dist.all_reduce(logits_max, op=dist.ReduceOp.MAX, group=mpu.get_tensor_model_parallel_group())
        normalized_vocab_parallel_logits = vocab_parallel_logits - logits_max
        normalized_exp_logits = normalized_vocab_parallel_logits.exp()
        normalized_sum_exp_logits = normalized_exp_logits.sum(dim=-1, keepdim=True)
        dist.all_reduce(normalized_sum_exp_logits, group=mpu.get_tensor_model_parallel_group())
        softmax_logits = normalized_exp_logits / normalized_sum_exp_logits
        sum_softmax_times_logits = (softmax_logits * vocab_parallel_logits).sum(dim=-1, keepdim=True)
        dist.all_reduce(sum_softmax_times_logits, group=mpu.get_tensor_model_parallel_group())
        entropy = logits_max + normalized_sum_exp_logits.log() - sum_softmax_times_logits
        ctx.save_for_backward(vocab_parallel_logits, softmax_logits, sum_softmax_times_logits)
        return entropy.squeeze(dim=-1)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        vocab_parallel_logits, softmax_logits, sum_softmax_times_logits = ctx.saved_tensors
        grad_input = grad_output.unsqueeze(dim=-1) * softmax_logits * (sum_softmax_times_logits - vocab_parallel_logits)
        return grad_input


def vocab_parallel_entropy(vocab_parallel_logits: torch.Tensor) -> torch.Tensor:
    """Compute entropy when the logits are sharded in tp ranks
    
    Args:
        vocab_parallel_logits: (total_nnz, vocab_size // tp_size)

    Returns: (total_nnz,)
        
    """
    return _VocabParallelEntropy.apply(vocab_parallel_logits)


def vocab_parallel_log_probs_from_logits(logits, labels):
    """TODO(zhangchi.usc1992): We may change the implementation later"""
    return -tensor_parallel.vocab_parallel_cross_entropy(vocab_parallel_logits=logits, target=labels)


def vocab_parallel_log_probs_from_logits_response_rmpad(input_ids, attention_mask, logits_rmpad, response_length):
    """Similar to log_probs_from_logits_response_rmpad, but the logits_rmpad is now spliited across tensor parallel region.
    This will further reduce the peak memory usage during training

    Args:
        input_ids: [batch_size, seqlen]
        attention_mask: [batch_size, seqlen]
        logits_rmpad: [total_nnz, vocab_size // tp_size]
        response_length: int

    """
    from flash_attn.bert_padding import pad_input, unpad_input

    batch_size, seqlen = input_ids.shape
    input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask=attention_mask)
    input_ids_rmpad = input_ids_rmpad.squeeze(-1)
    input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=0)
    full_log_probs_rmpad = vocab_parallel_log_probs_from_logits(logits=logits_rmpad,
                                                                labels=input_ids_rmpad_rolled)  # (total_nnz,)
    full_output = pad_input(hidden_states=full_log_probs_rmpad.unsqueeze(-1),
                            indices=indices,
                            batch=batch_size,
                            seqlen=seqlen)
    output = full_output.squeeze(-1)[:, -response_length - 1:-1]  # [batch_size, response_length]
    return output


def vocab_parallel_compute_entropy_loss(logits, eos_mask):
    """Compute Categorical entropy loss

    Args:
        logits: `(torch.Tensor)`
            shape: (bs, response_length, vocab_size)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        entropy: a scalar torch.Tensor

    """
    # compute entropy
    entropy = vocab_parallel_entropy(logits)
    entropy_loss = verl_F.masked_mean(entropy, mask=eos_mask)
    return entropy_loss
