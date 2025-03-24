"""Transformer."""
from contextlib import nullcontext
import os
import math
import numpy as np
import torch
from functools import wraps
import torch.nn.functional as F
from typing import Optional

from megatron import core
from apex.corex.activations import swiglu
from apex.transformer.functional import fused_apply_rotary_pos_emb
from megatron.training import get_timers, get_args, get_rlhf_args
from megatron.legacy.model.module import MegatronModule
from megatron.legacy.model.transformer import (
    ParallelMLP,
    CoreAttention,
    ParallelAttention,
    ParallelTransformerLayer,
    ParallelTransformer
)
from megatron.core import mpu, tensor_parallel
from megatron.core.enums import ModelType
from megatron.legacy.model.enums import AttnMaskType, LayerType, AttnType
from megatron.legacy.model.fused_softmax import FusedScaleMaskSoftmax
from megatron.legacy.model.fused_bias_gelu import bias_gelu_impl
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding, apply_rotary_pos_emb
from megatron.core.num_microbatches_calculator import get_num_microbatches
from megatron.legacy.model.utils import attention_mask_func, openai_gelu, erf_gelu, get_norm, get_rmsnorm_residual
from megatron.core.tensor_parallel import (
    gather_from_sequence_parallel_region_to_moe,
    reduce_scatter_to_sequence_parallel_region_from_moe,
    get_cuda_rng_tracker,
    get_data_parallel_rng_tracker_name
)
from megatron.core.parallel_state import get_tensor_model_parallel_group, get_tensor_and_expert_parallel_group
from megatron.core.jit import jit_fuser
from megatron.core import ixte_extensions
import megatron.legacy.model.transformer
from megatron.legacy.model.transformer import (
    DropPath,
    SwitchMLP,
    bias_dropout_add_fused_train,
    bias_dropout_add_fused_inference,
    get_bias_dropout_add,
    _get_num_layers,
    _get_layer_type,
    NoopTransformerLayer,
    FlashSelfAttention
)
from megatronspeed.core import parallel_state

import deepspeed
from deepspeed.moe.layer import MoE
from deepspeed.accelerator import get_accelerator

try:
    from deepspeed.sequence.layer import DistributedAttention
    dist_attn_supported = True
except ImportError:
    dist_attn_supported = False

try:
    from einops import rearrange
except ImportError:
    rearrange = None

try:
    from flash_attn.flash_attn_interface import flash_attn_unpadded_func
except ImportError:
    try:
        from flash_attn.flash_attn_interface import flash_attn_varlen_func as flash_attn_unpadded_func
    except ImportError:
        flash_attn_unpadded_func = None


def parallel_mlp_init(self, config, is_expert=False, moe=False, enable_expert_tensor_parallelism=False, rlhf_training=False):
    super(ParallelMLP, self).__init__()
    args = get_args()

    self.add_bias = config.add_bias_linear

    ffn_hidden_size = config.ffn_hidden_size
    if config.gated_linear_unit:
        ffn_hidden_size *= 2

    # Project to 4h. If using swiglu double the output width, see https://arxiv.org/pdf/2002.05202.pdf
    self.dense_h_to_4h = tensor_parallel.ColumnParallelLinear(
        config.hidden_size,
        ffn_hidden_size,
        config=config,
        init_method=config.init_method,
        bias=self.add_bias,
        gather_output=False,
        skip_bias_add=True,
        is_expert=is_expert,
        moe=moe,
        enable_expert_tensor_parallelism=enable_expert_tensor_parallelism
    )

    self.bias_gelu_fusion = False
    self.activation_func = None
    self.swiglu = args.swiglu

    if args.openai_gelu:
        self.activation_func = openai_gelu
    elif args.onnx_safe:
        self.activation_func = erf_gelu
    elif args.swiglu:
        # def swiglu(x):
        #     x = torch.chunk(x, 2, dim=-1)
        #     return F.silu(x[0]) * x[1]
        self.activation_func = swiglu
    elif args.squared_relu:
        def squared_relu(x):
            return torch.pow(F.relu(x), 2)
        self.activation_func = squared_relu
    else:
        self.bias_gelu_fusion = args.bias_gelu_fusion
        self.activation_func = F.gelu

    # Project back to h.
    self.dense_4h_to_h = tensor_parallel.RowParallelLinear(
        config.ffn_hidden_size,
        config.hidden_size,
        config=config,
        init_method=config.output_layer_init_method,
        bias=self.add_bias,
        skip_bias_add=True,
        input_is_parallel=True,
        is_expert=is_expert,
        moe=moe,
        enable_expert_tensor_parallelism=enable_expert_tensor_parallelism
    )

def parallel_mlp_forward_wrapper(fn):
    @wraps(fn)
    def wrapper(self, hidden_states, inference_params=None):
        args = get_args()

        # if not args.deepspeed:
        #     return fn(self, hidden_states)

        # [s, b, 4hp]
        intermediate_parallel, bias_parallel = self.dense_h_to_4h(hidden_states, inference_params=inference_params)

        if self.bias_gelu_fusion:
            assert self.add_bias is True
            # DeepSpeed FLOPS profiler temporarily substitues functions like F.gelu to calculate the throughput
            assert hasattr(self, "__flops__") or self.activation_func == F.gelu
            intermediate_parallel = bias_gelu_impl(intermediate_parallel, bias_parallel)
        else:
            if bias_parallel is not None:
                intermediate_parallel = intermediate_parallel + bias_parallel
            intermediate_parallel = self.activation_func(intermediate_parallel)

        # [s, b, h]
        output, output_bias = self.dense_4h_to_h(intermediate_parallel, inference_params=inference_params)
        return output, output_bias
    return wrapper

# class ParallelMLP(MegatronModule):
#     """MLP.

#     MLP will take the input with h hidden state, project it to 4*h
#     hidden dimension, perform nonlinear transformation, and project the
#     state back into h hidden dimension.
#     """

#     def __init__(self, config, is_expert=False, moe=False, enable_expert_tensor_parallelism=False):
#         super(ParallelMLP, self).__init__()
#         args = get_args()

#         self.add_bias = config.add_bias_linear

#         ffn_hidden_size = config.ffn_hidden_size
#         if config.gated_linear_unit:
#             ffn_hidden_size *= 2

#         # Project to 4h. If using swiglu double the output width, see https://arxiv.org/pdf/2002.05202.pdf
#         self.dense_h_to_4h = tensor_parallel.ColumnParallelLinear(
#             config.hidden_size,
#             ffn_hidden_size,
#             config=config,
#             init_method=config.init_method,
#             bias=self.add_bias,
#             gather_output=False,
#             skip_bias_add=True,
#             is_expert=is_expert,
#             moe=moe,
#             enable_expert_tensor_parallelism=enable_expert_tensor_parallelism
#         )

#         self.bias_gelu_fusion = False
#         self.activation_func = None
#         self.swiglu = args.swiglu

#         if args.openai_gelu:
#             self.activation_func = openai_gelu
#         elif args.onnx_safe:
#             self.activation_func = erf_gelu
#         elif args.swiglu:
#             def swiglu(x):
#                 x = torch.chunk(x, 2, dim=-1)
#                 return F.silu(x[0]) * x[1]
#             self.activation_func = swiglu
#         elif args.squared_relu:
#             def squared_relu(x):
#                 return torch.pow(F.relu(x), 2)
#             self.activation_func = squared_relu
#         else:
#             self.bias_gelu_fusion = args.bias_gelu_fusion
#             self.activation_func = F.gelu

#         # Project back to h.
#         self.dense_4h_to_h = tensor_parallel.RowParallelLinear(
#             config.ffn_hidden_size,
#             config.hidden_size,
#             config=config,
#             init_method=config.output_layer_init_method,
#             bias=self.add_bias,
#             skip_bias_add=True,
#             input_is_parallel=True,
#             is_expert=is_expert,
#             moe=moe,
#             enable_expert_tensor_parallelism=enable_expert_tensor_parallelism
#         )

#     def forward(self, hidden_states):

#         # [s, b, 4hp]
#         intermediate_parallel, bias_parallel = self.dense_h_to_4h(hidden_states)

#         if self.bias_gelu_fusion:
#             assert self.add_bias is True
#             # DeepSpeed FLOPS profiler temporarily substitues functions like F.gelu to calculate the throughput
#             assert hasattr(self, "__flops__") or self.activation_func == F.gelu
#             intermediate_parallel = bias_gelu_impl(intermediate_parallel, bias_parallel)
#         else:
#             if bias_parallel is not None:
#                 intermediate_parallel = intermediate_parallel + bias_parallel
#             intermediate_parallel = self.activation_func(intermediate_parallel)

#         # [s, b, h]
#         output, output_bias = self.dense_4h_to_h(intermediate_parallel)
#         return output, output_bias

def core_attention_init(self, layer_number, config,
                        attn_mask_type=AttnMaskType.padding):
    super(CoreAttention, self).__init__()
    self.fp16 = config.fp16
    self.bf16 = config.bf16

    self.apply_query_key_layer_scaling = config.apply_query_key_layer_scaling
    self.attention_softmax_in_fp32 = config.attention_softmax_in_fp32
    if self.apply_query_key_layer_scaling:
        self.attention_softmax_in_fp32 = True
    self.layer_number = max(1, layer_number)
    self.attn_mask_type = attn_mask_type
    self.sequence_parallel = config.sequence_parallel

    projection_size = config.kv_channels * config.num_attention_heads

    # Per attention head and per partition values.
    seq_parallel_world_size = 1
    if parallel_state.sequence_parallel_is_initialized():
        seq_parallel_world_size = mpu.get_tensor_model_parallel_world_size()
    world_size = seq_parallel_world_size if seq_parallel_world_size > 1 else mpu.get_tensor_model_parallel_world_size()
    self.hidden_size_per_partition = core.utils.divide(projection_size,
                                                        world_size)
    self.hidden_size_per_attention_head = core.utils.divide(
        projection_size, config.num_attention_heads)
    self.num_attention_heads_per_partition = core.utils.divide(
        config.num_attention_heads, world_size)

    coeff = None
    self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)
    if self.apply_query_key_layer_scaling:
        coeff = self.layer_number
        self.norm_factor *= coeff

    self.scale_mask_softmax = FusedScaleMaskSoftmax(
        self.fp16, self.bf16,
        self.attn_mask_type,
        config.masked_softmax_fusion,
        attention_mask_func,
        self.attention_softmax_in_fp32,
        coeff)

    # Dropout. Note that for a single iteration, this layer will generate
    # different outputs on different number of parallel partitions but
    # on average it should not be partition dependent.
    self.attention_dropout = torch.nn.Dropout(config.attention_dropout)

# class CoreAttention(MegatronModule):

#     def __init__(self, layer_number, config,
#                  attn_mask_type=AttnMaskType.padding):
#         super(CoreAttention, self).__init__()
#         self.fp16 = config.fp16
#         self.bf16 = config.bf16

#         self.apply_query_key_layer_scaling = config.apply_query_key_layer_scaling
#         self.attention_softmax_in_fp32 = config.attention_softmax_in_fp32
#         if self.apply_query_key_layer_scaling:
#             self.attention_softmax_in_fp32 = True
#         self.layer_number = max(1, layer_number)
#         self.attn_mask_type = attn_mask_type
#         self.sequence_parallel = config.sequence_parallel

#         projection_size = config.kv_channels * config.num_attention_heads

#         # Per attention head and per partition values.
#         seq_parallel_world_size = 1
#         if parallel_state.sequence_parallel_is_initialized():
#             seq_parallel_world_size = mpu.get_tensor_model_parallel_world_size()
#         world_size = seq_parallel_world_size if seq_parallel_world_size > 1 else mpu.get_tensor_model_parallel_world_size()
#         self.hidden_size_per_partition = core.utils.divide(projection_size,
#                                                            world_size)
#         self.hidden_size_per_attention_head = core.utils.divide(
#             projection_size, config.num_attention_heads)
#         self.num_attention_heads_per_partition = core.utils.divide(
#             config.num_attention_heads, world_size)

#         coeff = None
#         self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)
#         if self.apply_query_key_layer_scaling:
#             coeff = self.layer_number
#             self.norm_factor *= coeff

#         self.scale_mask_softmax = FusedScaleMaskSoftmax(
#             self.fp16, self.bf16,
#             self.attn_mask_type,
#             config.masked_softmax_fusion,
#             attention_mask_func,
#             self.attention_softmax_in_fp32,
#             coeff)

#         # Dropout. Note that for a single iteration, this layer will generate
#         # different outputs on different number of parallel partitions but
#         # on average it should not be partition dependent.
#         self.attention_dropout = torch.nn.Dropout(config.attention_dropout)

#     def forward(self, query_layer, key_layer,
#                 value_layer, attention_mask):

#         # ===================================
#         # Raw attention scores. [b, np, s, s]
#         # ===================================

#         # [b, np, sq, sk]
#         output_size = (query_layer.size(1),
#                        query_layer.size(2),
#                        query_layer.size(0),
#                        key_layer.size(0))

#         # [sq, b, np, hn] -> [sq, b * np, hn]
#         query_layer = query_layer.reshape(output_size[2],
#                                           output_size[0] * output_size[1], -1)
#         # [sk, b, np, hn] -> [sk, b * np, hn]
#         key_layer = key_layer.view(output_size[3],
#                                    output_size[0] * output_size[1], -1)

#         # preallocting input tensor: [b * np, sq, sk]
#         matmul_input_buffer = mpu.get_global_memory_buffer().get_tensor(
#             (output_size[0]*output_size[1], output_size[2], output_size[3]),
#             query_layer.dtype, "mpu")

#         # Raw attention scores. [b * np, sq, sk]
#         matmul_result = torch.baddbmm(
#             matmul_input_buffer,
#             query_layer.transpose(0, 1),   # [b * np, sq, hn]
#             key_layer.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
#             beta=0.0, alpha=(1.0/self.norm_factor))

#         # change view to [b, np, sq, sk]
#         attention_scores = matmul_result.view(*output_size)

#         # ===========================
#         # Attention probs and dropout
#         # ===========================

#         # attention scores and attention mask [b, np, sq, sk]
#         attention_probs = self.scale_mask_softmax(attention_scores,
#                                                   attention_mask)

#         # This is actually dropping out entire tokens to attend to, which might
#         # seem a bit unusual, but is taken from the original Transformer paper.
#         if not self.sequence_parallel:
#             with tensor_parallel.get_cuda_rng_tracker().fork():
#                 attention_probs = self.attention_dropout(attention_probs)
#         else:
#             attention_probs = self.attention_dropout(attention_probs)

#         # =========================
#         # Context layer. [sq, b, hp]
#         # =========================

#         # value_layer -> context layer.
#         # [sk, b, np, hn] --> [b, np, sq, hn]

#         # context layer shape: [b, np, sq, hn]
#         output_size = (value_layer.size(1),
#                        value_layer.size(2),
#                        query_layer.size(0),
#                        value_layer.size(3))

#         # change view [sk, b * np, hn]
#         value_layer = value_layer.view(value_layer.size(0),
#                                        output_size[0] * output_size[1], -1)

#         # change view [b * np, sq, sk]
#         attention_probs = attention_probs.view(output_size[0] * output_size[1],
#                                                output_size[2], -1)

#         # matmul: [b * np, sq, hn]
#         context_layer = torch.bmm(attention_probs, value_layer.transpose(0, 1))

#         # change view [b, np, sq, hn]
#         context_layer = context_layer.view(*output_size)

#         # [b, np, sq, hn] --> [sq, b, np, hn]
#         context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

#         # [sq, b, np, hn] --> [sq, b, hp]
#         new_context_layer_shape = context_layer.size()[:-2] + \
#             (self.hidden_size_per_partition,)
#         context_layer = context_layer.view(*new_context_layer_shape)

#         return context_layer

def flash_selfattention_forward_wrapper(fn):
    @wraps(fn)
    def wrapper(self, q, k, v):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            q, k, v: The tensor containing the query, key, and value. (B, S, H, D)
        """

        assert all((i.dtype in [torch.float16, torch.bfloat16] for i in (q,k,v)))
        assert all((i.is_cuda for i in (q,k,v)))

        batch_size, seqlen_q, num_heads = q.shape[0], q.shape[1], q.shape[2]
        seqlen_k = k.shape[1]

        q, k, v = [rearrange(x, 'b s ... -> (b s) ...') for x in [q, k, v]]
        cu_seqlens_q = torch.arange(0, (batch_size + 1) * seqlen_q, step=seqlen_q, dtype=torch.int32,
                                    device=q.device)

        if self.training:
            # during training q,k,v always have same seqlen
            assert seqlen_k == seqlen_q

            is_causal = self.causal
            cu_seqlens_k = cu_seqlens_q
            dropout_p = self.dropout_p
        else:
            # turn off FA causal mask after first inference autoregressive iteration
            # only on first autoregressive step q,k,v have same seqlen
            is_causal = seqlen_q == seqlen_k
            cu_seqlens_k = torch.arange(0, (batch_size + 1) * seqlen_k, step=seqlen_k, dtype=torch.int32,
                        device=q.device)
            dropout_p = 0

        output = flash_attn_unpadded_func(
            q, k, v, cu_seqlens_q, cu_seqlens_k, seqlen_q, seqlen_k,
            dropout_p,
            softmax_scale=self.softmax_scale, causal=is_causal,
            imp_mode=1 if batch_size * num_heads >= 32 else 0
        )

        output = rearrange(output, '(b s) ... -> b s ...', b=batch_size)
        return output
    return wrapper

def parallel_attention_init(self, config, layer_number,
                            attention_type=AttnType.self_attn,
                            attn_mask_type=AttnMaskType.padding,
                            rlhf_training=False):
    super(ParallelAttention, self).__init__()
    if rlhf_training:
        args = get_rlhf_args()
    else:
        args = get_args()
    self.layer_number = max(1, layer_number)
    self.attention_type = attention_type
    self.attn_mask_type = attn_mask_type
    self.params_dtype = config.params_dtype
    self.sequence_parallel = config.sequence_parallel
    self.config = config
    self.group_query_attention = args.group_query_attention
    self.num_query_groups = args.num_query_groups
    self.num_attention_heads = config.num_attention_heads
    self.num_key_value_heads = args.num_query_groups
    self.use_gqa = args.group_query_attention

    query_projection_size = config.kv_channels * config.num_attention_heads
    if self.group_query_attention:
        kv_projection_size = args.kv_channels * args.num_query_groups
    else:
        kv_projection_size = args.kv_channels * args.num_attention_heads

    self.use_flash_attn = args.use_flash_attn \
        and attention_type == AttnType.self_attn \
        and self.attn_mask_type == AttnMaskType.causal
    if self.use_flash_attn:
        if flash_attn_unpadded_func is None:
            raise ImportError('FlashAttention is not installed, please install with '
                                'pip install flash-attn')
        assert attention_type == AttnType.self_attn, ('FlashAttention code path only supports '
                                                        'self-attention for now')
        assert self.attn_mask_type == AttnMaskType.causal, ('FlashAttention code path only '
                                                            'supports causal mask for now')
        if rearrange is None:
            raise ImportError('einops is not installed, please install with pip install einops')

    # Per attention head and per partition values.
    world_size = mpu.get_tensor_model_parallel_world_size()
    self.hidden_size_per_attention_head = core.utils.divide(
        query_projection_size, config.num_attention_heads)
    self.num_attention_heads_per_partition = core.utils.divide(
        config.num_attention_heads, world_size)

    if self.group_query_attention:
        if args.num_query_groups % world_size != 0:
            raise NotImplementedError('Currently the num_query_groups should be '
                                        'a multiple of the tensor parallel size')
        self.num_query_groups_per_partition = core.utils.divide(
                    args.num_query_groups, world_size)
    else:
        self.num_query_groups_per_partition = self.num_attention_heads_per_partition

    # Strided linear layer.
    if attention_type == AttnType.self_attn:
        self.query_key_value = tensor_parallel.ColumnParallelLinear(
            config.hidden_size,
            query_projection_size + 2 * kv_projection_size,
            config=config,
            init_method=config.init_method,
            bias=args.add_bias_linear or args.add_qkv_bias,
            gather_output=False)
    else:
        assert attention_type == AttnType.cross_attn

        if self.group_query_attention:
            raise NotImplementedError("Grouped query attention not implemented for cross-attention.")
        assert query_projection_size == kv_projection_size

        self.query = tensor_parallel.ColumnParallelLinear(
            config.hidden_size,
            query_projection_size,
            config=config,
            init_method=config.init_method,
            bias=config.add_bias_linear,
            gather_output=False)

        self.key_value = tensor_parallel.ColumnParallelLinear(
            config.hidden_size,
            2 * kv_projection_size,
            config=config,
            init_method=config.init_method,
            bias=config.add_bias_linear,
            gather_output=False)

    # Currently FlashAttention only works with causal mask
    if self.use_flash_attn:
        local_attn = FlashSelfAttention(causal=True, attention_dropout=config.attention_dropout)
    else:
        local_attn = CoreAttention(self.layer_number, config, self.attn_mask_type)

    self.enable_ds_sequence_parallel = False

    if args.deepspeed:
        self.enable_ds_sequence_parallel = parallel_state.get_sequence_parallel_world_size() > 1 \
                                            or args.force_ds_sequence_parallel

    if self.enable_ds_sequence_parallel:
        assert dist_attn_supported, 'Distributed attention is not supported in this DeepSpeed version'
        assert args.num_attention_heads % parallel_state.get_sequence_parallel_world_size() == 0
        self.dist_attn = DistributedAttention(
            local_attn, 
            parallel_state.get_sequence_parallel_group(), 
            gather_idx=1 if args.use_flash_attn else 0) 
        # flash_attn_cuda assumes [b, s, nh, hd] layout, we need to make sure all2all gathers into the correct sequence dimension.
    else:
        if self.use_flash_attn:
            self.core_attention_flash = local_attn
        else:
            self.core_attention = local_attn
            self.checkpoint_core_attention = config.recompute_granularity == 'selective'

    # Output.
    self.dense = tensor_parallel.RowParallelLinear(
        query_projection_size,
        config.hidden_size,
        config=config,
        init_method=config.output_layer_init_method,
        bias=args.add_bias_linear,
        input_is_parallel=True,
        skip_bias_add=True)

def parallel_attention_forward_wrapper(fn):
    @wraps(fn)
    def wrapper(self, hidden_states, attention_mask,
                encoder_output=None, inference_params=None,
                rotary_pos_emb=None, position_ids=None):
        # hidden_states: [sq, b, h]

        # Inference or Forward 使用, 会影响 RoPE
        if position_ids is not None:
            # position_ids = position_ids.transpose(1, 0) #[s, b]
            ## 适配rope fused kernel
            position_ids = position_ids.transpose(1, 0)[:, 0].unsqueeze(-1) #[s, b] -> [s, b] -> [s, 1]  rope position ids embedding 在同一位置是一样的

        # =================================================
        # Pre-allocate memory for key-values for inference.
        # =================================================
        is_first_step = False
        if inference_params:
            if self.layer_number not in inference_params.key_value_memory_dict:
                inf_max_seq_len = inference_params.max_sequence_length
                inf_max_batch_size = inference_params.max_batch_size
                inference_key_memory = self._allocate_memory(
                    inf_max_seq_len, inf_max_batch_size,
                    self.num_query_groups_per_partition)
                inference_value_memory = self._allocate_memory(
                    inf_max_seq_len, inf_max_batch_size,
                    self.num_query_groups_per_partition)

                inference_params.key_value_memory_dict[self.layer_number] = (
                    inference_key_memory, inference_value_memory)
                is_first_step = True
            else:
                inference_key_memory, inference_value_memory = \
                    inference_params.key_value_memory_dict[self.layer_number]

            # 存储 inference position_ids
            if is_first_step and position_ids is not None \
                    and "position_ids" not in inference_params.key_value_memory_dict:
                inference_params.key_value_memory_dict["position_ids"] = position_ids

        # =====================
        # Query, Key, and Value
        # =====================

        if self.attention_type == AttnType.self_attn:
            # Attention heads [sq, b, h] --> [sq, b, ng * (np/ng + 2) * hn)]
            mixed_x_layer, _ = self.query_key_value(hidden_states, inference_params=inference_params)

            # [sq, b, hp] --> [sq, b, ng, (np/ng + 2) * hn]
            new_tensor_shape = mixed_x_layer.size()[:-1] + (
                self.num_query_groups_per_partition,
                (
                    (self.num_attention_heads_per_partition // self.num_query_groups_per_partition + 2)
                    * self.hidden_size_per_attention_head
                ),
            )
            mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

            # [sq, b, ng, (np/ng + 2) * hn] --> [sq, b, ng, np/ng * hn], [sq, b, ng, hn], [sq, b, ng, hn]
            (query_layer,
            key_layer,
            value_layer) = torch.split(
                mixed_x_layer,
                [
                    (
                        self.num_attention_heads_per_partition // self.num_query_groups_per_partition
                        * self.hidden_size_per_attention_head
                    ),
                    self.hidden_size_per_attention_head,
                    self.hidden_size_per_attention_head
                ],
                dim=3)

            # [sq, b, ng, np/ng * hn] -> [sq, b, np, hn] -
            query_layer = query_layer.contiguous().view(query_layer.size(0), query_layer.size(1), -1, self.hidden_size_per_attention_head)
        else:
            # Attention heads [sk, b, h] --> [sk, b, (np * 2 * hn)]
            mixed_kv_layer, _ = self.key_value(encoder_output)

            # [sk, b, (np * 2 * hn)] --> [sk, b, np, 2 * hn]
            new_tensor_shape = mixed_kv_layer.size()[:-1] + \
                (self.num_attention_heads_per_partition,
                    2 * self.hidden_size_per_attention_head)
            mixed_kv_layer = mixed_kv_layer.view(*new_tensor_shape)

            # [sk, b, np, 2 * hn] --> 2 [sk, b, np, hn]
            (key_layer,
                value_layer) = tensor_parallel.split_tensor_along_last_dim(mixed_kv_layer, 2)

            # Attention head [sq, b, h] --> [sq, b, hp]
            query_layer, _ = self.query(hidden_states)
            # [sq, b, hp] --> [sq, b, np, hn]
            new_tensor_shape = query_layer.size()[:-1] + \
                (self.num_attention_heads_per_partition,
                    self.hidden_size_per_attention_head)
            query_layer = query_layer.view(*new_tensor_shape)

        # ==================================
        # Adjust key and value for inference
        # ==================================

        # duplicate the pos_emb for self attention
        if rotary_pos_emb is not None:
            if isinstance(rotary_pos_emb, tuple):
                rotary_pos_emb = rotary_pos_emb
            else:
                rotary_pos_emb = ((rotary_pos_emb,) * 2)

        if inference_params:
            batch_start = inference_params.batch_size_offset
            batch_end = batch_start + key_layer.size(1)
            assert batch_end <= inference_key_memory.size(1)
            sequence_start = inference_params.sequence_len_offset
            sequence_end = sequence_start + key_layer.size(0)
            assert sequence_end <= inference_key_memory.size(0)
            # Copy key and values.
            inference_key_memory[sequence_start:sequence_end,
                                    batch_start:batch_end, ...] = key_layer
            inference_value_memory[sequence_start:sequence_end,
                                    batch_start:batch_end, ...] = value_layer
            key_layer = inference_key_memory[
                :sequence_end, batch_start:batch_end, ...]
            value_layer = inference_value_memory[
                :sequence_end, batch_start:batch_end, ...]


            # adjust the key rotary positional embedding
            if rotary_pos_emb is not None:
                q_pos_emb, k_pos_emb = rotary_pos_emb
                # need to cross check this condition during inference
                # if not set_inference_key_value_memory:
                if not is_first_step:
                    # In inference, we compute one token at a time.
                    # Select the correct query positional embedding (only the last token in the sequence)
                    if position_ids is not None:
                        # 取 last position_id 对应的 q_pos_emb
                        assert position_ids.shape[0] == 1
                        # cur_pos_id = position_ids[-1].item()
                        q_pos_emb = q_pos_emb[position_ids].squeeze(2) # [1, bs, 1, dim]

                        # 取 position_id 对应的 k_pos_emb
                        k_pos_emb = k_pos_emb.squeeze(1).squeeze(1) # [max_seq, dim]
                        mem_position_ids = inference_params.key_value_memory_dict["position_ids"]
                        if mem_position_ids.shape[0] == sequence_end:
                            k_pos_emb = k_pos_emb[mem_position_ids].unsqueeze(2) # [sequence_end, b, 1, dim]
                        elif mem_position_ids.shape[0] == sequence_end - 1:
                            new_position_ids = torch.concat((mem_position_ids, position_ids), 0)
                            k_pos_emb = k_pos_emb[new_position_ids].unsqueeze(2) # [sequence_end, b, 1, dim]
                            inference_params.key_value_memory_dict["position_ids"] = new_position_ids # update memory position_ids
                        else:
                            raise Exception("input position_ids shape wrong.")
                    else:
                        q_pos_emb = q_pos_emb[sequence_end - 1 : sequence_end] # [1, 1, 1, dim]
                        k_pos_emb = k_pos_emb[:sequence_end, :, :, :] # [sequence_end, 1, 1, dim]
                else:
                    # In the first forward pass of inference, we use the entire provided prefix.
                    # q_pos_emb here has the rope embeddings of the entire prefix + to-be-generated output
                    # so we slice to just the prefix.
                    if position_ids is not None:
                        assert position_ids.shape[0] <= q_pos_emb.shape[0] and q_pos_emb.shape[0] == k_pos_emb.shape[0]
                        q_pos_emb = q_pos_emb.squeeze(1).squeeze(1) # [max_seq, dim]
                        q_pos_emb = q_pos_emb[position_ids].unsqueeze(2) # [s, b, 1, dim]
                        k_pos_emb = k_pos_emb.squeeze(1).squeeze(1) # [max_seq, dim]
                        k_pos_emb = k_pos_emb[position_ids].unsqueeze(2) # [s, b, 1, dim]
                    else:
                        q_pos_emb = q_pos_emb[:sequence_end, :, :, :] # [sequence_end, 1, 1, dim]
                        k_pos_emb = k_pos_emb[:sequence_end, :, :, :] # [sequence_end, 1, 1, dim]

                rotary_pos_emb = (q_pos_emb, k_pos_emb)


        # ==================================
        # core attention computation
        # ==================================

        # expand the key_layer and value_layer [sk, b, ng, hn] -> [sk, b, np, hn]
        # Flash attention support group attention
        if self.num_attention_heads_per_partition // self.num_query_groups_per_partition > 1 and not self.use_flash_attn:
            key_layer = key_layer.repeat_interleave(
                self.num_attention_heads_per_partition // self.num_query_groups_per_partition,
                dim = 2
            )
            value_layer = value_layer.repeat_interleave(
                self.num_attention_heads_per_partition // self.num_query_groups_per_partition,
                dim = 2
            )

        # apply relative positional encoding (rotary embedding)
        if rotary_pos_emb is not None:
            q_pos_emb, k_pos_emb = rotary_pos_emb
            # query_layer = apply_rotary_pos_emb(query_layer, q_pos_emb,self.config)
            # key_layer = apply_rotary_pos_emb(key_layer, k_pos_emb,self.config)
            query_layer = fused_apply_rotary_pos_emb(query_layer, q_pos_emb)
            key_layer = fused_apply_rotary_pos_emb(key_layer, k_pos_emb)
            # TODO, can apply positional embedding to value_layer so it has
            # absolute positional embedding.
            # otherwise, only relative positional embedding takes effect
            # value_layer = apply_rotary_pos_emb(value_layer, k_pos_emb)

        if self.enable_ds_sequence_parallel:
            batch_dim_idx = 1
            if self.use_flash_attn:
                query_layer, key_layer, value_layer = [rearrange(x, 's b ... -> b s ...').contiguous()
                        for x in (query_layer, key_layer, value_layer)]
                batch_dim_idx = 0

                context_layer = self.dist_attn(query_layer, key_layer, value_layer, batch_dim_idx)

                context_layer = rearrange(context_layer, 'b s h d -> s b (h d)').contiguous()
            else:
                context_layer = self.dist_attn(query_layer, key_layer, value_layer, attention_mask)
        else:
            if not self.use_flash_attn:
                if self.checkpoint_core_attention:
                    context_layer = self._checkpointed_attention_forward(
                        query_layer, key_layer, value_layer, attention_mask)
                else:
                    context_layer = self.core_attention(
                        query_layer, key_layer, value_layer, attention_mask)
            else:
                q, k, v = [rearrange(x, 's b ... -> b s ...').contiguous()
                        for x in (query_layer, key_layer, value_layer)]
                if not self.sequence_parallel:
                    with tensor_parallel.get_cuda_rng_tracker().fork():
                        context_layer = self.core_attention_flash(q, k, v)
                else:
                    context_layer = self.core_attention_flash(q, k, v)
                context_layer = rearrange(context_layer, 'b s h d -> s b (h d)').contiguous()

        # =================
        # Output. [sq, b, h]
        # =================

        output, bias = self.dense(context_layer, inference_params=inference_params)

        return output, bias
    return wrapper

# class ParallelAttention(MegatronModule):
#     """Parallel self-attention layer abstract class.

#     Self-attention layer takes input with size [s, b, h]
#     and returns output of the same size.
#     """

#     def __init__(self, config, layer_number,
#                  attention_type=AttnType.self_attn,
#                  attn_mask_type=AttnMaskType.padding):
#         super(ParallelAttention, self).__init__()
#         args = get_args()
#         self.layer_number = max(1, layer_number)
#         self.attention_type = attention_type
#         self.attn_mask_type = attn_mask_type
#         self.params_dtype = config.params_dtype
#         self.sequence_parallel = config.sequence_parallel
#         self.config = config
#         self.group_query_attention = args.group_query_attention
#         self.num_query_groups = args.num_query_groups
#         self.num_attention_heads = config.num_attention_heads
#         self.num_key_value_heads = args.num_query_groups
#         self.use_gqa = args.group_query_attention

#         query_projection_size = config.kv_channels * config.num_attention_heads
#         if self.group_query_attention:
#             kv_projection_size = args.kv_channels * args.num_query_groups
#         else:
#             kv_projection_size = args.kv_channels * args.num_attention_heads

#         self.use_flash_attn = args.use_flash_attn \
#             and attention_type == AttnType.self_attn \
#             and self.attn_mask_type == AttnMaskType.causal
#         if self.use_flash_attn:
#             if flash_attn_unpadded_func is None:
#                 raise ImportError('FlashAttention is not installed, please install with '
#                                   'pip install flash-attn')
#             assert attention_type == AttnType.self_attn, ('FlashAttention code path only supports '
#                                                           'self-attention for now')
#             assert self.attn_mask_type == AttnMaskType.causal, ('FlashAttention code path only '
#                                                                 'supports causal mask for now')
#             if rearrange is None:
#                 raise ImportError('einops is not installed, please install with pip install einops')

#         # Per attention head and per partition values.
#         world_size = mpu.get_tensor_model_parallel_world_size()
#         self.hidden_size_per_attention_head = core.utils.divide(
#             query_projection_size, config.num_attention_heads)
#         self.num_attention_heads_per_partition = core.utils.divide(
#             config.num_attention_heads, world_size)

#         if self.group_query_attention:
#             if args.num_query_groups % world_size != 0:
#                 raise NotImplementedError('Currently the num_query_groups should be '
#                                           'a multiple of the tensor parallel size')
#             self.num_query_groups_per_partition = core.utils.divide(
#                         args.num_query_groups, world_size)
#         else:
#             self.num_query_groups_per_partition = self.num_attention_heads_per_partition

#         # Strided linear layer.
#         if attention_type == AttnType.self_attn:
#             self.query_key_value = tensor_parallel.ColumnParallelLinear(
#                 config.hidden_size,
#                 query_projection_size + 2 * kv_projection_size,
#                 config=config,
#                 init_method=config.init_method,
#                 bias=args.add_bias_linear or args.add_qkv_bias,
#                 gather_output=False)
#         else:
#             assert attention_type == AttnType.cross_attn

#             if self.group_query_attention:
#                 raise NotImplementedError("Grouped query attention not implemented for cross-attention.")
#             assert query_projection_size == kv_projection_size

#             self.query = tensor_parallel.ColumnParallelLinear(
#                 config.hidden_size,
#                 query_projection_size,
#                 config=config,
#                 init_method=config.init_method,
#                 bias=config.add_bias_linear,
#                 gather_output=False)

#             self.key_value = tensor_parallel.ColumnParallelLinear(
#                 config.hidden_size,
#                 2 * kv_projection_size,
#                 config=config,
#                 init_method=config.init_method,
#                 bias=config.add_bias_linear,
#                 gather_output=False)

#         # Currently FlashAttention only works with causal mask
#         if self.use_flash_attn:
#             local_attn = FlashSelfAttention(causal=True, attention_dropout=config.attention_dropout)
#         else:
#             local_attn = CoreAttention(self.layer_number, config, self.attn_mask_type)

#         self.enable_ds_sequence_parallel = parallel_state.get_sequence_parallel_world_size() > 1 \
#                                            or args.force_ds_sequence_parallel
#         if self.enable_ds_sequence_parallel:
#             assert dist_attn_supported, 'Distributed attention is not supported in this DeepSpeed version'
#             assert args.num_attention_heads % parallel_state.get_sequence_parallel_world_size() == 0
#             self.dist_attn = DistributedAttention(
#                 local_attn, 
#                 parallel_state.get_sequence_parallel_group(), 
#                 gather_idx=1 if args.use_flash_attn else 0) 
#             # flash_attn_cuda assumes [b, s, nh, hd] layout, we need to make sure all2all gathers into the correct sequence dimension.
#         else:
#             if self.use_flash_attn:
#                 self.core_attention_flash = local_attn
#             else:
#                 self.core_attention = local_attn
#                 self.checkpoint_core_attention = config.recompute_granularity == 'selective'

#         # Output.
#         self.dense = tensor_parallel.RowParallelLinear(
#             query_projection_size,
#             config.hidden_size,
#             config=config,
#             init_method=config.output_layer_init_method,
#             bias=args.add_bias_linear,
#             input_is_parallel=True,
#             skip_bias_add=True)


#     def _checkpointed_attention_forward(self, query_layer, key_layer,
#                                         value_layer, attention_mask,
#                                         rotary_pos_emb=None):
#         """Forward method with activation checkpointing."""
#         def custom_forward(*inputs):
#             query_layer = inputs[0]
#             key_layer = inputs[1]
#             value_layer = inputs[2]
#             attention_mask = inputs[3]
#             output_ = self.core_attention(query_layer, key_layer,
#                                           value_layer, attention_mask)
#             return output_

#         q_pos_emb, k_pos_emb = (None, None) if rotary_pos_emb is None \
#             else rotary_pos_emb

#         hidden_states = tensor_parallel.checkpoint(
#             custom_forward,
#             False, query_layer, key_layer, value_layer, attention_mask,
#             q_pos_emb, k_pos_emb)

#         return hidden_states

#     def _allocate_memory(self, inference_max_sequence_len, batch_size, num_attention_heads):
#         return torch.empty(
#             inference_max_sequence_len,
#             batch_size,
#             num_attention_heads,
#             self.hidden_size_per_attention_head,
#             dtype=self.params_dtype,
#             device=torch.cuda.current_device())

#     def repeat_kv(self, hidden_states, n_rep):
#         slen, batch, num_key_value_heads_per_partition, head_dim = hidden_states.shape
#         if n_rep == 1:
#             return hidden_states
#         elif num_key_value_heads_per_partition == 1:
#             # If no of KV heads is 1 then just perform expand operation
#             # instead of unsqueeze, expand and reshape to match query states.
#             return hidden_states.expand(slen, batch, n_rep, head_dim)
#         else:
#             hidden_states = hidden_states[:, :, :, None, :].expand(
#                 slen, batch, num_key_value_heads_per_partition, n_rep, head_dim)
#             return hidden_states.reshape(slen, batch,
#                                          num_key_value_heads_per_partition * n_rep,
#                                          head_dim)
                                     
#     def split_tensor(self, mixed_x_layer):
#         query_layer, key_layer, value_layer = torch.split(mixed_x_layer, [self.num_key_value_groups, 1, 1], dim=-2)
#         query_layer = query_layer.reshape(mixed_x_layer.shape[:2] + (self.num_attention_heads_per_partition, self.hidden_size_per_attention_head))
#         key_layer = torch.squeeze(key_layer, -2)
#         value_layer = torch.squeeze(value_layer, -2)

#         return query_layer, key_layer, value_layer

#     def forward(self, hidden_states, attention_mask,
#                 encoder_output=None, inference_params=None,
#                 rotary_pos_emb=None):
#         # hidden_states: [sq, b, h]

#         # =================================================
#         # Pre-allocate memory for key-values for inference.
#         # =================================================
#         is_first_step = False
#         if inference_params:
#             if self.layer_number not in inference_params.key_value_memory_dict:
#                 inf_max_seq_len = inference_params.max_sequence_length
#                 inf_max_batch_size = inference_params.max_batch_size
#                 inference_key_memory = self._allocate_memory(
#                     inf_max_seq_len, inf_max_batch_size,
#                     self.num_query_groups_per_partition)
#                 inference_value_memory = self._allocate_memory(
#                     inf_max_seq_len, inf_max_batch_size,
#                     self.num_query_groups_per_partition)

#                 inference_params.key_value_memory_dict[self.layer_number] = (
#                     inference_key_memory, inference_value_memory)
#                 is_first_step = True
#             else:
#                 inference_key_memory, inference_value_memory = \
#                     inference_params.key_value_memory_dict[self.layer_number]

#         # =====================
#         # Query, Key, and Value
#         # =====================

#         if self.attention_type == AttnType.self_attn:
#             # Attention heads [sq, b, h] --> [sq, b, ng * (np/ng + 2) * hn)]
#             mixed_x_layer, _ = self.query_key_value(hidden_states)

#             # [sq, b, hp] --> [sq, b, ng, (np/ng + 2) * hn]
#             new_tensor_shape = mixed_x_layer.size()[:-1] + (
#                 self.num_query_groups_per_partition,
#                 (
#                     (self.num_attention_heads_per_partition // self.num_query_groups_per_partition + 2)
#                     * self.hidden_size_per_attention_head
#                 ),
#             )
#             mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

#             # [sq, b, ng, (np/ng + 2) * hn] --> [sq, b, ng, np/ng * hn], [sq, b, ng, hn], [sq, b, ng, hn]
#             (query_layer,
#             key_layer,
#             value_layer) = torch.split(
#                 mixed_x_layer,
#                 [
#                     (
#                         self.num_attention_heads_per_partition // self.num_query_groups_per_partition
#                         * self.hidden_size_per_attention_head
#                     ),
#                     self.hidden_size_per_attention_head,
#                     self.hidden_size_per_attention_head
#                 ],
#                 dim=3)

#             # [sq, b, ng, np/ng * hn] -> [sq, b, np, hn] -
#             query_layer = query_layer.reshape(query_layer.size(0), query_layer.size(1), -1, self.hidden_size_per_attention_head)
#         else:
#             # Attention heads [sk, b, h] --> [sk, b, (np * 2 * hn)]
#             mixed_kv_layer, _ = self.key_value(encoder_output)

#             # [sk, b, (np * 2 * hn)] --> [sk, b, np, 2 * hn]
#             new_tensor_shape = mixed_kv_layer.size()[:-1] + \
#                 (self.num_attention_heads_per_partition,
#                  2 * self.hidden_size_per_attention_head)
#             mixed_kv_layer = mixed_kv_layer.view(*new_tensor_shape)

#             # [sk, b, np, 2 * hn] --> 2 [sk, b, np, hn]
#             (key_layer,
#              value_layer) = tensor_parallel.split_tensor_along_last_dim(mixed_kv_layer, 2)

#             # Attention head [sq, b, h] --> [sq, b, hp]
#             query_layer, _ = self.query(hidden_states)
#             # [sq, b, hp] --> [sq, b, np, hn]
#             new_tensor_shape = query_layer.size()[:-1] + \
#                 (self.num_attention_heads_per_partition,
#                  self.hidden_size_per_attention_head)
#             query_layer = query_layer.view(*new_tensor_shape)

#         # ==================================
#         # Adjust key and value for inference
#         # ==================================

#         # duplicate the pos_emb for self attention
#         if rotary_pos_emb is not None:
#             if isinstance(rotary_pos_emb, tuple):
#                 rotary_pos_emb = rotary_pos_emb
#             else:
#                 rotary_pos_emb = ((rotary_pos_emb,) * 2)

#         if inference_params:
#             batch_start = inference_params.batch_size_offset
#             batch_end = batch_start + key_layer.size(1)
#             assert batch_end <= inference_key_memory.size(1)
#             sequence_start = inference_params.sequence_len_offset
#             sequence_end = sequence_start + key_layer.size(0)
#             assert sequence_end <= inference_key_memory.size(0)
#             # Copy key and values.
#             inference_key_memory[sequence_start:sequence_end,
#                                  batch_start:batch_end, ...] = key_layer
#             inference_value_memory[sequence_start:sequence_end,
#                                    batch_start:batch_end, ...] = value_layer
#             key_layer = inference_key_memory[
#                 :sequence_end, batch_start:batch_end, ...]
#             value_layer = inference_value_memory[
#                 :sequence_end, batch_start:batch_end, ...]


#             # adjust the key rotary positional embedding
#             if rotary_pos_emb is not None:
#                 q_pos_emb, k_pos_emb = rotary_pos_emb
#                 # need to cross check this condition during inference
#                 # if not set_inference_key_value_memory:
#                 if not is_first_step:
#                     # In inference, we compute one token at a time.
#                     # Select the correct positional embedding
#                     # (only the last token in the sequence)
#                     q_pos_emb = q_pos_emb[sequence_end - 1 : sequence_end]
#                 else:
#                     # In the first forward pass of inference,
#                     # we use the entire provided prefix.
#                     # q_pos_emb here has the rope embeddings of the entire
#                     # prefix + to-be-generated output so
#                     # we slice to just the prefix.
#                     q_pos_emb = q_pos_emb[:sequence_end, :, :, :]
#                 k_pos_emb = k_pos_emb[:sequence_end, :, :, :]
#                 rotary_pos_emb = (q_pos_emb, k_pos_emb)


#         # ==================================
#         # core attention computation
#         # ==================================

#         # expand the key_layer and value_layer [sk, b, ng, hn] -> [sk, b, np, hn]
#         # Flash attention support group attention
#         if self.num_attention_heads_per_partition // self.num_query_groups_per_partition > 1 and not self.use_flash_attn:
#             key_layer = key_layer.repeat_interleave(
#                 self.num_attention_heads_per_partition // self.num_query_groups_per_partition,
#                 dim = 2
#             )
#             value_layer = value_layer.repeat_interleave(
#                 self.num_attention_heads_per_partition // self.num_query_groups_per_partition,
#                 dim = 2
#             )

#         # apply relative positional encoding (rotary embedding)
#         if rotary_pos_emb is not None:
#             q_pos_emb, k_pos_emb = rotary_pos_emb
#             query_layer = apply_rotary_pos_emb(query_layer, q_pos_emb,self.config)
#             key_layer = apply_rotary_pos_emb(key_layer, k_pos_emb,self.config)
#             # TODO, can apply positional embedding to value_layer so it has
#             # absolute positional embedding.
#             # otherwise, only relative positional embedding takes effect
#             # value_layer = apply_rotary_pos_emb(value_layer, k_pos_emb)

#         if self.enable_ds_sequence_parallel:
#             batch_dim_idx = 1
#             if self.use_flash_attn:
#                 query_layer, key_layer, value_layer = [rearrange(x, 's b ... -> b s ...').contiguous()
#                         for x in (query_layer, key_layer, value_layer)]
#                 batch_dim_idx = 0

#                 context_layer = self.dist_attn(query_layer, key_layer, value_layer, batch_dim_idx)

#                 context_layer = rearrange(context_layer, 'b s h d -> s b (h d)').contiguous()
#             else:
#                 context_layer = self.dist_attn(query_layer, key_layer, value_layer, attention_mask)
#         else:
#             if not self.use_flash_attn:
#                 if self.checkpoint_core_attention:
#                     context_layer = self._checkpointed_attention_forward(
#                         query_layer, key_layer, value_layer, attention_mask)
#                 else:
#                     context_layer = self.core_attention(
#                         query_layer, key_layer, value_layer, attention_mask)
#             else:
#                 q, k, v = [rearrange(x, 's b ... -> b s ...').contiguous()
#                         for x in (query_layer, key_layer, value_layer)]
#                 if not self.sequence_parallel:
#                     with tensor_parallel.get_cuda_rng_tracker().fork():
#                         context_layer = self.core_attention_flash(q, k, v)
#                 else:
#                     context_layer = self.core_attention_flash(q, k, v)
#                 context_layer = rearrange(context_layer, 'b s h d -> s b (h d)').contiguous()

#         # =================
#         # Output. [sq, b, h]
#         # =================

#         output, bias = self.dense(context_layer)

#         return output, bias

def parallel_transformer_layer_init(self, config,
                                    layer_number, layer_type=LayerType.encoder,
                                    self_attn_mask_type=AttnMaskType.padding,
                                    drop_path_rate=0., num_experts=1,
                                    rlhf_training=False):
    if rlhf_training:
        args = get_rlhf_args()
    else:
        args = get_args()
    self.args = args

    super(ParallelTransformerLayer, self).__init__()
    self.layer_number = layer_number
    self.layer_type = layer_type

    self.apply_residual_connection_post_norm \
        = config.apply_residual_connection_post_layernorm

    self.bf16 = config.bf16
    self.fp32_residual_connection = config.fp32_residual_connection

    # Normalize the input data.
    self.input_norm = get_norm(config)

    # Self attention.
    self.self_attention = ParallelAttention(
        config,
        layer_number,
        attention_type=AttnType.self_attn,
        attn_mask_type=self_attn_mask_type,
        rlhf_training=rlhf_training)
    self.hidden_dropout = config.hidden_dropout
    self.bias_dropout_fusion = config.bias_dropout_fusion
    self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else None

    # Normalize the attention output
    if self.args.normalization != "RMSNorm":
        self.post_attention_norm = get_norm(config)
    else:
        self.post_attention_norm = get_rmsnorm_residual(config)

    # Cross attention.
    if self.layer_type in (LayerType.decoder,
                            LayerType.retro_decoder,
                            LayerType.retro_decoder_with_retriever,
                            LayerType.retro_encoder):
        self.inter_attention = ParallelAttention(
            config,
            layer_number,
            attention_type=AttnType.cross_attn,
            rlhf_training=rlhf_training)
        # Normalize the attention output.
        self.post_inter_attention_norm = get_norm(config)

    # MLP
    self.num_experts = num_experts
    if not args.deepspeed:
        if args.num_experts is not None:
            self.mlp = SwitchMLP(config) # Megatron-LM's MoE
        else:
            self.mlp = ParallelMLP(config, rlhf_training=rlhf_training)
    else:
        if self.num_experts <= 1: # dense, not MoE
            self.mlp = ParallelMLP(config, rlhf_training=rlhf_training)
        else: # DeepSpeed's MoE
            enable_expert_tensor_parallelism = args.enable_expert_tensor_parallelism
            self.mlp = MoE(args.hidden_size,
                            ParallelMLP(config,
                                        moe=True,
                                        enable_expert_tensor_parallelism=enable_expert_tensor_parallelism),
                            num_experts=self.num_experts,
                            ep_size=args.moe_expert_parallel_size,
                            k=args.topk,
                            use_residual=(args.mlp_type == 'residual'),
                            capacity_factor=args.moe_train_capacity_factor,
                            eval_capacity_factor=args.moe_eval_capacity_factor,
                            min_capacity=args.moe_min_capacity,
                            drop_tokens=args.moe_token_dropping,
                            use_tutel=args.use_tutel,
                            enable_expert_tensor_parallelism=enable_expert_tensor_parallelism,
                            top2_2nd_expert_sampling=args.moe_top2_2nd_expert_sampling)

    # Set bias+dropout+add fusion grad_enable execution handler.
    TORCH_MAJOR = int(torch.__version__.split('.')[0])
    TORCH_MINOR = int(torch.__version__.split('.')[1])
    use_nvfuser = TORCH_MAJOR > 1 or (TORCH_MAJOR == 1 and TORCH_MINOR >= 10)
    self.bias_dropout_add_exec_handler = \
            nullcontext if use_nvfuser else torch.enable_grad

    if args.retro_add_retriever:
        self.retro_num_neighbors = args.retro_num_neighbors
        self.retro_chunk_length = args.retro_chunk_length
        self.retro_retrieved_length = \
            args.retro_num_retrieved_chunks * args.retro_chunk_length

    # Retriever (bi-directional transformer with cross attention)
    if layer_type == LayerType.retro_decoder_with_retriever:
        self.retriever = ParallelTransformer(
            config=config,
            model_type=ModelType.retro_encoder,
            self_attn_mask_type=AttnMaskType.padding,
            pre_process=True,
            post_process=False,
        )
        self._retriever_key = 'retriever'
    else:
        self.retriever = None

def parallel_transformer_layer_forward_wrapper(fn):
    @wraps(fn)
    def wrapper(self, hidden_states, attention_mask=None,
                encoder_output=None, enc_dec_attn_mask=None,
                retriever_input=None,
                retriever_output=None,
                retriever_attn_mask=None,
                inference_params=None,
                rotary_pos_emb=None,
                position_ids=None, **kwargs):
        # Update the params in case the retro param changes during inference
        # TODO: better redesign with inference param
        args = get_args()

        # if not args.deepspeed:
        #     return fn(self, hidden_states, attention_mask=attention_mask,
        #               encoder_output=encoder_output, enc_dec_attn_mask=enc_dec_attn_mask,
        #               retriever_input=retriever_input,
        #               retriever_output=retriever_output,
        #               retriever_attn_mask=retriever_attn_mask,
        #               inference_params=inference_params,
        #               rotary_pos_emb=rotary_pos_emb, **kwargs)

        if args.retro_add_retriever:
            self.retro_num_neighbors = args.retro_num_neighbors
            self.retro_chunk_length = args.retro_chunk_length
            self.retro_retrieved_length = \
                args.retro_num_retrieved_chunks * args.retro_chunk_length

        # hidden_states: [s, b, h]

        # Layer norm at the beginning of the transformer layer.
        norm_output = self.input_norm(hidden_states)

        # Self attention.
        attention_output, attention_bias = \
            self.self_attention(
                norm_output,
                attention_mask,
                inference_params=inference_params,
                rotary_pos_emb=rotary_pos_emb,
                position_ids=position_ids,)

        # Residual connection.
        if self.apply_residual_connection_post_norm:
            residual = norm_output
        else:
            residual = hidden_states

        if self.drop_path is None:
            # jit scripting for a nn.module (with dropout) is not
            # trigerring the fusion kernel. For now, we use two
            # different nn.functional routines to account for varying
            # dropout semantics during training and inference phases.
            if self.bias_dropout_fusion:
                if self.training:
                    bias_dropout_add_func = bias_dropout_add_fused_train
                else:
                    bias_dropout_add_func = bias_dropout_add_fused_inference
            else:
                bias_dropout_add_func = get_bias_dropout_add(self.training)

            if attention_bias is not None:
                attention_bias = attention_bias.expand_as(residual)
            with self.bias_dropout_add_exec_handler():
            #     norm_input = bias_dropout_add_func(
            #         attention_output,
            #         attention_bias,
            #         residual,
            #         self.hidden_dropout)
                if self.args.normalization != "RMSNorm":
                    norm_input = bias_dropout_add_func(
                        attention_output,
                        attention_bias,
                        residual,
                        self.hidden_dropout)
                else:
                    if attention_bias is not None:
                        attention_output = attention_output + attention_bias
                    out = torch.nn.functional.dropout(attention_output, p=self.hidden_dropout, training=self.training)
                    norm_output, norm_input = self.post_attention_norm(out, residual)
        else:
            out = torch.nn.functional.dropout(attention_output + attention_bias,
                                                p=self.hidden_dropout,
                                                training=self.training)
            # norm_input = residual + self.drop_path(out)
            if self.args.normalization != "RMSNorm":
                norm_input = residual + self.drop_path(out)
            else:
                norm_output, norm_input = self.post_attention_norm(self.drop_path(out), residual)

        # Layer norm post the self attention.
        # norm_output = self.post_attention_norm(norm_input)
        if self.args.normalization != "RMSNorm":
            norm_output = self.post_attention_norm(norm_input)

        # Cross attention.
        if self.layer_type == LayerType.encoder:
            pass
        elif self.layer_type == LayerType.decoder:
            norm_input, norm_output = \
                self.default_decoder_cross_attention(
                    encoder_output,
                    enc_dec_attn_mask,
                    norm_input,
                    norm_output,
                    bias_dropout_add_func)
        elif self.layer_type == LayerType.retro_encoder:
            norm_input, norm_output = \
                self.retro_encoder_cross_attention(
                    retriever_output,
                    norm_input,
                    norm_output,
                    bias_dropout_add_func)
        elif self.layer_type in (LayerType.retro_decoder,
                                    LayerType.retro_decoder_with_retriever):
            retriever_output, norm_input, norm_output = \
                self.retro_decoder_cross_attention(
                    retriever_input,
                    retriever_output,
                    retriever_attn_mask,
                    norm_input,
                    norm_output,
                    inference_params,
                    bias_dropout_add_func)
        else:
            raise Exception("Unsupported layer type, '%s'." %
                            self.layer_type.name)

        # MLP.
        moe_loss = torch.tensor(0.0, device=norm_output.device, dtype=norm_output.dtype)
        mlp_bias = torch.tensor(0.0, device=norm_output.device, dtype=norm_output.dtype)

        if self.num_experts > 1 and args.deepspeed:
            mlp_output, moe_loss, _ = self.mlp(norm_output)
        else:
            mlp_output, mlp_bias = self.mlp(norm_output, inference_params=inference_params)

        # Second residual connection.
        if self.apply_residual_connection_post_norm:
            residual = norm_output
        else:
            residual = norm_input

        if self.drop_path is None:
            if mlp_bias is not None:
                mlp_bias = mlp_bias.expand_as(residual)
            with self.bias_dropout_add_exec_handler():
                output = bias_dropout_add_func(
                    mlp_output,
                    mlp_bias,
                    residual,
                    self.hidden_dropout)

            # Jit compiled function creates 'view' tensor. This tensor
            # potentially gets saved in the MPU checkpoint function context,
            # which rejects view tensors. While making a viewless tensor here
            # won't result in memory savings (like the data loader, or
            # p2p_communication), it serves to document the origin of this
            # 'view' tensor.
            output = core.utils.make_viewless_tensor(inp = output,
                                                        requires_grad = output.requires_grad,
                                                        keep_graph = True)

        else:
            if mlp_bias is not None:
                mlp_output = mlp_output + mlp_bias
            out = torch.nn.functional.dropout(mlp_output,
                                                p=self.hidden_dropout,
                                                training=self.training)
            output = residual + self.drop_path(out)

        if args.deepspeed:
            if self.layer_type == LayerType.retro_decoder_with_retriever:
                return output, retriever_output, moe_loss
            else:
                return output, moe_loss
        else:
            if self.layer_type == LayerType.retro_decoder_with_retriever:
                return output, retriever_output
            else:
                return output
    return wrapper

# class ParallelTransformerLayer(MegatronModule):
#     """A single transformer layer.

#     Transformer layer takes input with size [s, b, h] and returns an
#     output of the same size.
#     """

#     def __init__(self, config,
#                  layer_number, layer_type=LayerType.encoder,
#                  self_attn_mask_type=AttnMaskType.padding,
#                  drop_path_rate=0., num_experts=1):
#         args = get_args()

#         super(ParallelTransformerLayer, self).__init__()
#         self.layer_number = layer_number
#         self.layer_type = layer_type

#         self.apply_residual_connection_post_norm \
#             = config.apply_residual_connection_post_layernorm

#         self.bf16 = config.bf16
#         self.fp32_residual_connection = config.fp32_residual_connection

#         # Normalize the input data.
#         self.input_norm = get_norm(config)

#         # Self attention.
#         self.self_attention = ParallelAttention(
#             config,
#             layer_number,
#             attention_type=AttnType.self_attn,
#             attn_mask_type=self_attn_mask_type)
#         self.hidden_dropout = config.hidden_dropout
#         self.bias_dropout_fusion = config.bias_dropout_fusion
#         self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else None

#         # Normalize the attention output
#         self.post_attention_norm = get_norm(config)

#         # Cross attention.
#         if self.layer_type in (LayerType.decoder,
#                                LayerType.retro_decoder,
#                                LayerType.retro_decoder_with_retriever,
#                                LayerType.retro_encoder):
#             self.inter_attention = ParallelAttention(
#                 config,
#                 layer_number,
#                 attention_type=AttnType.cross_attn)
#             # Normalize the attention output.
#             self.post_inter_attention_norm = get_norm(config)

#         # MLP
#         self.num_experts = num_experts
#         if not args.deepspeed:
#             if args.num_experts is not None:
#                 self.mlp = SwitchMLP(config) # Megatron-LM's MoE
#             else:
#                 self.mlp = ParallelMLP(config)
#         else:
#             if self.num_experts <= 1: # dense, not MoE
#                 self.mlp = ParallelMLP(config)
#             else: # DeepSpeed's MoE
#                 enable_expert_tensor_parallelism = args.enable_expert_tensor_parallelism
#                 self.mlp = MoE(args.hidden_size,
#                                ParallelMLP(config,
#                                            moe=True,
#                                            enable_expert_tensor_parallelism=enable_expert_tensor_parallelism),
#                                num_experts=self.num_experts,
#                                ep_size=args.moe_expert_parallel_size,
#                                k=args.topk,
#                                use_residual=(args.mlp_type == 'residual'),
#                                capacity_factor=args.moe_train_capacity_factor,
#                                eval_capacity_factor=args.moe_eval_capacity_factor,
#                                min_capacity=args.moe_min_capacity,
#                                drop_tokens=args.moe_token_dropping,
#                                use_tutel=args.use_tutel,
#                                enable_expert_tensor_parallelism=enable_expert_tensor_parallelism,
#                                top2_2nd_expert_sampling=args.moe_top2_2nd_expert_sampling)

#         # Set bias+dropout+add fusion grad_enable execution handler.
#         TORCH_MAJOR = int(torch.__version__.split('.')[0])
#         TORCH_MINOR = int(torch.__version__.split('.')[1])
#         use_nvfuser = TORCH_MAJOR > 1 or (TORCH_MAJOR == 1 and TORCH_MINOR >= 10)
#         self.bias_dropout_add_exec_handler = \
#                 nullcontext if use_nvfuser else torch.enable_grad

#         if args.retro_add_retriever:
#             self.retro_num_neighbors = args.retro_num_neighbors
#             self.retro_chunk_length = args.retro_chunk_length
#             self.retro_retrieved_length = \
#                 args.retro_num_retrieved_chunks * args.retro_chunk_length

#         # Retriever (bi-directional transformer with cross attention)
#         if layer_type == LayerType.retro_decoder_with_retriever:
#             self.retriever = ParallelTransformer(
#                 config=config,
#                 model_type=ModelType.retro_encoder,
#                 self_attn_mask_type=AttnMaskType.padding,
#                 pre_process=True,
#                 post_process=False,
#             )
#             self._retriever_key = 'retriever'
#         else:
#             self.retriever = None

#     def default_decoder_cross_attention(self,
#                                         encoder_output,
#                                         enc_dec_attn_mask,
#                                         norm_input,
#                                         norm_output,
#                                         bias_dropout_add_func):
#         '''Cross attention for a standard encoder-decoder model.'''

#         # Attention.
#         attention_output, attention_bias = \
#             self.inter_attention(norm_output,
#                                  enc_dec_attn_mask,
#                                  encoder_output=encoder_output)

#         # Residual connection.
#         if self.apply_residual_connection_post_norm:
#             residual = norm_output
#         else:
#             residual = norm_input

#         if attention_bias is not None:
#             attention_bias = attention_bias.expand_as(residual)

#         # Bias-dropout-add.
#         with self.bias_dropout_add_exec_handler():
#             norm_input = bias_dropout_add_func(
#                 attention_output,
#                 attention_bias,
#                 residual,
#                 self.hidden_dropout)

#         # Normalize.
#         norm_output = self.post_inter_attention_norm(norm_input)

#         return norm_input, norm_output

#     def retro_encoder_cross_attention(self,
#                                       retriever_output,
#                                       norm_input,
#                                       norm_output,
#                                       bias_dropout_add_func):
#         """Cross attention for Retro encoder.

#         Notation:
#             ns : Sequence length.
#             bs : Batch size.
#             d  : Hidden size.
#             l  : Number of chunks per sample (i.e., seq_length/chunk_length).
#             k  : Number of neighbors.
#             r  : Number of retrieved tokens (neighbors + continuation).
#         """

#         ns, bs, d = norm_output.shape # [r, bs * l * k, d]

#         # Divide sequence dimension into chunks.
#         chunked_outputs = norm_output.reshape(self.retro_retrieved_length,
#                                               -1,
#                                               self.retro_num_neighbors,
#                                               d)
#         chunked_outputs_before_norm = \
#             norm_input.reshape(self.retro_retrieved_length, -1,
#                                self.retro_num_neighbors, d) # [r, bs*l, k, d]

#         # Per-chunk attention.
#         norm_inputs = []
#         norm_outputs = []
#         for k in range(self.retro_num_neighbors):

#             # Attention.
#             chunked_output = chunked_outputs[:,:,k].contiguous()
#             attention_output, attention_bias = \
#                 self.inter_attention(
#                     chunked_output, # Q (neighbor embedding)
#                     None,
#                     encoder_output=retriever_output) # K, V (hidden act)

#             # Residual connection.
#             if self.apply_residual_connection_post_norm:
#                 residual = chunked_output
#             else:
#                 residual = chunked_outputs_before_norm[:,:,k]

#             # Re-enable torch grad to enable fused optimization.
#             with torch.enable_grad():
#                 norm_input = bias_dropout_add_func(
#                     attention_output,
#                     None if attention_bias is None else attention_bias.expand_as(residual),
#                     residual,
#                     self.hidden_dropout)
#                 norm_inputs.append(norm_input)

#             # Layer norm.
#             norm_output = self.post_inter_attention_norm(norm_input)
#             norm_outputs.append(norm_output)

#         # Concatenate layer norms.
#         # norm_input : [r, k * bs * l, d]
#         # norm_output : [r, k * bs * l, d]
#         norm_input = torch.stack(norm_inputs, dim=1).reshape(ns, bs, d)
#         norm_output = torch.stack(norm_outputs, dim=1).reshape(ns, bs, d)

#         return norm_input, norm_output

#     def retro_decoder_cross_attention(self,
#                                       retriever_input,
#                                       retriever_output,
#                                       retriever_attn_mask,
#                                       norm_input,
#                                       norm_output,
#                                       inference_params,
#                                       bias_dropout_add_func):
#         """Cross attention for Retro decoder.

#         Notation:
#             ns : Sequence length.
#             bs : Batch size.
#             d  : Hidden size.
#             l  : Number of chunks per sample (i.e., seq_length/chunk_length).
#             m  : Number of tokens per chunk.
#             k  : Number of neighbors.
#             r  : Number of retrieved tokens (neighbors + continuation).
#         """

#         ns, bs, d = norm_output.shape
#         l = int(np.ceil(ns / self.retro_chunk_length))

#         # Retrieve neighbors.
#         if self.layer_type == LayerType.retro_decoder_with_retriever:
#             first_ns = ns % self.retro_chunk_length
#             if first_ns > 0:
#                 first_chunk, rest_chunk = \
#                     norm_output[:first_ns], norm_output[first_ns:]
#                 first_chunk = torch.nn.functional.pad(
#                     first_chunk,
#                     (0, 0, 0, 0, 0, self.retro_chunk_length - first_ns),
#                     'constant',
#                     0)
#                 chunked_output = \
#                     torch.cat((first_chunk, rest_chunk), dim=0) # [l * m, bs, d]
#             else:
#                 chunked_output = norm_output # [l * m, bs, d]
#             chunked_output = chunked_output \
#                 .reshape(l, self.retro_chunk_length, bs, d) \
#                 .permute(1, 2, 0, 3) \
#                 .reshape(self.retro_chunk_length, bs * l, d) \
#                 .contiguous()

#             # Get Encoder Output
#             retriever_output = self.retriever(
#                 hidden_states=retriever_input,
#                 attention_mask=retriever_attn_mask,
#                 retriever_output=chunked_output,
#                 retriever_attn_mask=retriever_attn_mask,
#                 inference_params=inference_params) # [r, k * bs * l , d]
#             retriever_output = retriever_output.reshape(
#                 self.retro_retrieved_length * self.retro_num_neighbors, bs * l, d) # [r * k, bs * l, d]

#         # Chunks.
#         pad = (ns - 1) % self.retro_chunk_length
#         attending_chunks = norm_output[pad:]
#         padded_chunks = torch.nn.functional.pad(
#             attending_chunks,
#             (0, 0, 0, 0, 0, self.retro_chunk_length - 1),
#             'constant', 0)
#         padded_chunked_output = padded_chunks \
#             .reshape(l, self.retro_chunk_length, bs, d) \
#             .permute(1, 2, 0, 3)
#         padded_chunked_output = padded_chunked_output.reshape(
#             self.retro_chunk_length, bs * l, d).contiguous()

#         # Encoder output.
#         attention_output, attention_bias = \
#             self.inter_attention(padded_chunked_output,
#                                  None,
#                                  encoder_output=retriever_output)

#         # Residual connection.
#         if self.apply_residual_connection_post_norm:
#             residual = norm_output
#         else:
#             residual = norm_input

#         # Re-enable torch grad to enable fused optimization.
#         with torch.enable_grad():
#             norm_input = bias_dropout_add_func(
#                 attention_output,
#                 None if attention_bias is None else attention_bias.expand_as(attention_output),
#                 torch.zeros_like(attention_output),
#                 self.hidden_dropout)
#             norm_input = norm_input \
#                 .reshape(self.retro_chunk_length, bs, l, d) \
#                 .permute(2, 0, 1, 3) # [l, m, bs, d]
#             norm_input = norm_input.reshape(self.retro_chunk_length * l, bs, d)
#             norm_input = torch.nn.functional.pad(
#                 norm_input,
#                 (0, 0, 0, 0, pad, 0),
#                 'constant', 0)[:ns] # [ns, b, d]
#             # TODO: better redesign with inference param
#             args = get_args()
#             norm_input = args.retro_attention_gate * norm_input + residual

#         # Layer norm post the decoder attention
#         norm_output = self.post_inter_attention_norm(norm_input)

#         return retriever_output, norm_input, norm_output

#     def forward(self, hidden_states, attention_mask=None,
#                 encoder_output=None, enc_dec_attn_mask=None,
#                 retriever_input=None,
#                 retriever_output=None,
#                 retriever_attn_mask=None,
#                 inference_params=None,
#                 rotary_pos_emb=None, **kwargs):

#         # Update the params in case the retro param changes during inference
#         # TODO: better redesign with inference param
#         args = get_args()
#         if args.retro_add_retriever:
#             self.retro_num_neighbors = args.retro_num_neighbors
#             self.retro_chunk_length = args.retro_chunk_length
#             self.retro_retrieved_length = \
#                 args.retro_num_retrieved_chunks * args.retro_chunk_length

#         # hidden_states: [s, b, h]

#         # Layer norm at the beginning of the transformer layer.
#         norm_output = self.input_norm(hidden_states)

#         # Self attention.
#         attention_output, attention_bias = \
#             self.self_attention(
#                 norm_output,
#                 attention_mask,
#                 inference_params=inference_params,
#                 rotary_pos_emb=rotary_pos_emb)

#         # Residual connection.
#         if self.apply_residual_connection_post_norm:
#             residual = norm_output
#         else:
#             residual = hidden_states

#         if self.drop_path is None:
#             # jit scripting for a nn.module (with dropout) is not
#             # trigerring the fusion kernel. For now, we use two
#             # different nn.functional routines to account for varying
#             # dropout semantics during training and inference phases.
#             if self.bias_dropout_fusion:
#                 if self.training:
#                     bias_dropout_add_func = bias_dropout_add_fused_train
#                 else:
#                     bias_dropout_add_func = bias_dropout_add_fused_inference
#             else:
#                 bias_dropout_add_func = get_bias_dropout_add(self.training)

#             if attention_bias is not None:
#                 attention_bias = attention_bias.expand_as(residual)
#             with self.bias_dropout_add_exec_handler():
#                 norm_input = bias_dropout_add_func(
#                     attention_output,
#                     attention_bias,
#                     residual,
#                     self.hidden_dropout)
#         else:
#             out = torch.nn.functional.dropout(attention_output + attention_bias,
#                                               p=self.hidden_dropout,
#                                               training=self.training)
#             norm_input = residual + self.drop_path(out)

#         # Layer norm post the self attention.
#         norm_output = self.post_attention_norm(norm_input)

#         # Cross attention.
#         if self.layer_type == LayerType.encoder:
#             pass
#         elif self.layer_type == LayerType.decoder:
#             norm_input, norm_output = \
#                 self.default_decoder_cross_attention(
#                     encoder_output,
#                     enc_dec_attn_mask,
#                     norm_input,
#                     norm_output,
#                     bias_dropout_add_func)
#         elif self.layer_type == LayerType.retro_encoder:
#             norm_input, norm_output = \
#                 self.retro_encoder_cross_attention(
#                     retriever_output,
#                     norm_input,
#                     norm_output,
#                     bias_dropout_add_func)
#         elif self.layer_type in (LayerType.retro_decoder,
#                                  LayerType.retro_decoder_with_retriever):
#             retriever_output, norm_input, norm_output = \
#                 self.retro_decoder_cross_attention(
#                     retriever_input,
#                     retriever_output,
#                     retriever_attn_mask,
#                     norm_input,
#                     norm_output,
#                     inference_params,
#                     bias_dropout_add_func)
#         else:
#             raise Exception("Unsupported layer type, '%s'." %
#                             self.layer_type.name)

#         # MLP.
#         moe_loss = torch.tensor(0.0, device=norm_output.device, dtype=norm_output.dtype)
#         mlp_bias = torch.tensor(0.0, device=norm_output.device, dtype=norm_output.dtype)

#         if self.num_experts > 1 and args.deepspeed:
#             mlp_output, moe_loss, _ = self.mlp(norm_output)
#         else:
#             mlp_output, mlp_bias = self.mlp(norm_output)

#         # Second residual connection.
#         if self.apply_residual_connection_post_norm:
#             residual = norm_output
#         else:
#             residual = norm_input

#         if self.drop_path is None:
#             if mlp_bias is not None:
#                 mlp_bias = mlp_bias.expand_as(residual)
#             with self.bias_dropout_add_exec_handler():
#                 output = bias_dropout_add_func(
#                     mlp_output,
#                     mlp_bias,
#                     residual,
#                     self.hidden_dropout)

#             # Jit compiled function creates 'view' tensor. This tensor
#             # potentially gets saved in the MPU checkpoint function context,
#             # which rejects view tensors. While making a viewless tensor here
#             # won't result in memory savings (like the data loader, or
#             # p2p_communication), it serves to document the origin of this
#             # 'view' tensor.
#             output = core.utils.make_viewless_tensor(inp = output,
#                                                      requires_grad = output.requires_grad,
#                                                      keep_graph = True)

#         else:
#             if mlp_bias is not None:
#                 mlp_output = mlp_output + mlp_bias
#             out = torch.nn.functional.dropout(mlp_output,
#                                               p=self.hidden_dropout,
#                                               training=self.training)
#             output = residual + self.drop_path(out)

#         if self.layer_type == LayerType.retro_decoder_with_retriever:
#             return output, retriever_output, moe_loss
#         else:
#             return output, moe_loss


class ParallelTransformerLayerPipe(ParallelTransformerLayer):
    """Extends ParallelTransformerLayer to forward attention_mask through the pipeline.

    Forward has two usages that affect attention mask communication:

    1) forward((input, attn_mask) , **kwargs) -> (output, mask)
       When the attention mask is provided as the second positional
       argument, typical pipeline behavior is used and both the output
       *and* mask are returned in a tuple. This tuple is then forwarded
       to the next stage in the pipeline.

       This version is useful if masks are dynamic.

    2) forward(input, **kwargs) -> output
       When the mask is static over all samples, it is advantageous to
       cache the mask and avoid communicating it.

       If no mask is provided, the module will query `self._args.attn_mask`
       for the mask and only return `super().forward(...)`
    """
    def __init__(self, config,
                 layer_number, layer_type=LayerType.encoder,
                 self_attn_mask_type=AttnMaskType.padding,
                 drop_path_rate=0., num_experts=1,
                 input_aggregated_moe_loss=False, return_aggregated_moe_loss=False):
        self.input_aggregated_moe_loss = input_aggregated_moe_loss
        self.return_aggregated_moe_loss = return_aggregated_moe_loss
        super().__init__(config, layer_number, layer_type, self_attn_mask_type, drop_path_rate, num_experts)

    def forward(self, inputs, **kwargs):
        assert torch.is_tensor(inputs) or isinstance(inputs, tuple)
        if not hasattr(self, '_args'):
            self._args = get_args()
        rotary_pos_emb = self._args.rotary_pos_emb if self._args.use_rotary_position_embeddings else None
        if torch.is_tensor(inputs) or len(inputs) == 1:
            assert not self.input_aggregated_moe_loss, f'Expecting an input tuple of size >= 2'
            # No attention mask forwarded, search for args.attn_mask
            hidden_states, attention_mask = inputs, self._args.attn_mask
            output, moe_loss = super().forward(hidden_states, attention_mask, **kwargs, rotary_pos_emb=rotary_pos_emb)
            return (output, moe_loss) if self.return_aggregated_moe_loss else output
        elif len(inputs) in (2, 3):
            # Attention mask and aggregated_moe can both be activations.
            return_attention_mask = False
            if len(inputs) == 2:
                if self.input_aggregated_moe_loss:
                    hidden_states, aggregated_moe_loss = inputs[0], inputs[1]
                    attention_mask = self._args.attn_mask
                else:
                    hidden_states, attention_mask = inputs[0], inputs[1]
                    return_attention_mask = True
            else:
                hidden_states, attention_mask, aggregated_moe_loss = inputs[0], inputs[1], inputs[2]

            # Forward aggregated_moe_loss to ParallelTransformerLayer for further accumulation
            if self.input_aggregated_moe_loss:
                kwargs.update({'aggregated_moe_loss': aggregated_moe_loss})

            output, moe_loss = super().forward(hidden_states, attention_mask, **kwargs, rotary_pos_emb=rotary_pos_emb)

            ret = (output, )
            if return_attention_mask:
                ret += (attention_mask, )
            if self.return_aggregated_moe_loss:
                ret += (moe_loss, )
            return ret
        else:
            raise RuntimeError('Received more inputs than understood.')

def get_num_experts_per_layer(num_experts: list, num_layers: int, expert_interval: int, offset: int = 0) -> list:
    assert len(num_experts) == 1 or len(num_experts) == num_layers // expert_interval, \
        'num_experts must be either a single value or a list of the same length as the number of MoE layers'
    if len(num_experts) == 1:
        num_experts = num_experts * (num_layers // expert_interval)
    experts_per_layer = []
    for i in range(num_layers):
        layer_num = i + 1 + offset
        n_e = num_experts[(layer_num-1) // expert_interval] if layer_num % expert_interval == 0 else 1
        experts_per_layer.append(n_e)
    return experts_per_layer

def parallel_transformer_init(self, config,
                              model_type, layer_type=LayerType.encoder,
                              self_attn_mask_type=AttnMaskType.padding,
                              post_norm=True,
                              pre_process=True,
                              post_process=True,
                              drop_path_rate=0.0,
                              rlhf_training=False):
    super(ParallelTransformer, self).__init__()
    if rlhf_training:
        args = get_rlhf_args()
    else:
        args = get_args()

    self.layer_type = layer_type
    self.model_type = model_type
    self.bf16 = config.bf16
    self.fp32_residual_connection = config.fp32_residual_connection
    self.post_norm = post_norm
    self.pre_process = pre_process
    self.post_process = post_process
    self.input_tensor = None
    self.drop_path_rate = drop_path_rate
    self.transformer_impl = args.transformer_impl
    self.retro_add_retriever = args.retro_add_retriever
    self.ds_inference = args.ds_inference
    self.deepspeed = args.deepspeed

    # Store activation checkpoiting flag.
    self.checkpoint_activations = args.checkpoint_activations
    self.checkpoint_num_layers = args.checkpoint_num_layers
    self.recompute_granularity = config.recompute_granularity
    if args.recompute_method_per_stage != None:
        if args.virtual_pipeline_model_parallel_size != None:
            if args.recompute_method_per_stage[mpu.get_virtual_pipeline_model_parallel_rank() * args.pipeline_model_parallel_size + mpu.get_pipeline_model_parallel_rank()] == 0:
                self.recompute_method = 'uniform'
            elif args.recompute_method_per_stage[mpu.get_virtual_pipeline_model_parallel_rank() * args.pipeline_model_parallel_size + mpu.get_pipeline_model_parallel_rank()] == 1:
                self.recompute_method = 'block'
        else:
            if args.recompute_method_per_stage[mpu.get_pipeline_model_parallel_rank()] == 0:
                self.recompute_method = 'uniform'
            elif args.recompute_method_per_stage[mpu.get_pipeline_model_parallel_rank()] == 1:
                self.recompute_method = 'block'
    else:
        self.recompute_method = config.recompute_method
    if args.recompute_num_layers_per_stage != None:
        if args.virtual_pipeline_model_parallel_size != None:
            self.recompute_num_layers = args.recompute_num_layers_per_stage[mpu.get_virtual_pipeline_model_parallel_rank() * args.pipeline_model_parallel_size + mpu.get_pipeline_model_parallel_rank()]
        else:
            self.recompute_num_layers = args.recompute_num_layers_per_stage[mpu.get_pipeline_model_parallel_rank()]
    else:
        self.recompute_num_layers = config.recompute_num_layers
    self.distribute_saved_activations = \
        config.distribute_saved_activations and not config.sequence_parallel

    self.sequence_parallel = config.sequence_parallel

    # Transformer Engine Init.
    self.transformer_engine_rope_available = False
    self.transformer_engine_v_0_10 = False
    self.transformer_engine_v_0_11 = False
    self.transformer_engine_v_0_8 = False
    self.ixte_v_0_2_3 = False
    self.use_ixte = False
    if self.transformer_impl == 'transformer_engine':
        global transformer_engine
        import transformer_engine
        megatron.legacy.model.transformer.transformer_engine = transformer_engine
        from importlib.metadata import version
        from pkg_resources import packaging

        if ixte_extensions._USE_IXTE:
            te_version = packaging.version.Version(ixte_extensions.te_version())
            self.use_ixte = True
            ixte_version = packaging.version.Version(ixte_extensions.ixte_version())
            if ixte_version >= packaging.version.Version("0.2.3"):
                self.ixte_v_0_2_3 = True
        else:
            te_version = packaging.version.Version(version("transformer-engine"))
        if te_version >= packaging.version.Version("0.8.0"):
            self.transformer_engine_v_0_8 = True
        if te_version >= packaging.version.Version("0.10.0"):
            self.transformer_engine_v_0_10 = True
        if te_version >= packaging.version.Version("0.11.0"):
            self.transformer_engine_v_0_11 = True
        if te_version >= packaging.version.Version("0.10.0"):
            self.transformer_engine_rope_available = True

        del version, packaging

        assert not args.squared_relu, "TransformerEngine does not support squared relu activation."

    self.use_fp8 = args.fp8 is not None
    self.fp8_recipe = None
    self.fp8_group = None
    if self.use_fp8:
        assert args.transformer_impl == 'transformer_engine', \
            'transformer-engine required for fp8 training and inference'
        self.fp8_group = mpu.get_amax_reduction_group()
        if args.fp8 == "e4m3":
            fp8_format = transformer_engine.common.recipe.Format.E4M3
        elif args.fp8 == "hybrid":
            fp8_format = transformer_engine.common.recipe.Format.HYBRID
        else:
            raise ValueError("The DelayedScaling recipe only supports E4M3 and HYBRID formats.")
        self.fp8_recipe = transformer_engine.common.recipe.DelayedScaling(
            margin=args.fp8_margin,
            interval=args.fp8_interval,
            fp8_format=fp8_format,
            amax_history_len=args.fp8_amax_history_len,
            amax_compute_algo=args.fp8_amax_compute_algo,
            override_linear_precision=(False, False, not args.fp8_wgrad),
        )

    self.num_microbatches_in_previous_step = -1
    self.microbatch_count = 0
    self.checkpoint_core_attention = config.recompute_granularity == 'selective'

    ## check custom parition pp stage
    if args.num_layers_per_stage is not None:
        assert sum(args.num_layers_per_stage) == args.num_layers, \
            f"total custom partition pp stage transformer layers should equal to model layers" \
            f"get total custom partition layers ({sum(args.num_layers_per_stage)})  !=   model layers ({args.num_layers})"

    # Number of layers.
    self.num_layers = _get_num_layers(args, model_type,
                                        layer_type==LayerType.decoder)

    self.drop_path_rates = [
        rate.item() for rate in
        torch.linspace(0, self.drop_path_rate, config.num_layers)]

    self.retro_layer_numbers = None
    if model_type == ModelType.retro_decoder:
        retro_layer_start = 6 if config.num_layers <= 15 else 9
        self.retro_layer_numbers = \
            np.arange(retro_layer_start, args.num_layers + 1, 3).tolist()
    if model_type == ModelType.retro_encoder:
        self.retro_layer_numbers = [1]

    # Transformer layers.
    if args.retro_add_retriever:
        assert self.recompute_granularity != 'full', \
            "Full recompute not supported for Retro."
        assert args.transformer_impl == 'local', \
            "Transformer engine does not support Retro layers."
    def build_layer(layer_number, n_e=1):
        if args.transformer_impl == 'local':
            current_layer_type = _get_layer_type(
                model_type, layer_type, self.retro_layer_numbers,
                layer_number)
            return ParallelTransformerLayer(
                config,
                layer_number,
                layer_type=current_layer_type,
                self_attn_mask_type=self_attn_mask_type,
                drop_path_rate=self.drop_path_rates[layer_number - 1],
                num_experts=n_e,
                rlhf_training=rlhf_training)
        else:
            # This argument is only available from TE v0.10 onwards.
            extra_transformer_engine_kwargs = {}
            if self.transformer_engine_v_0_8:
                extra_transformer_engine_kwargs["bias"] = args.add_bias_linear
            if self.transformer_engine_v_0_10:
                extra_transformer_engine_kwargs["activation"] = "swiglu" if args.swiglu else "gelu"
            if self.transformer_engine_v_0_11:
                extra_transformer_engine_kwargs["normalization"] = args.normalization
            if not ixte_extensions._USE_IXTE:
                assert config.attention_softmax_in_fp32, "TransformerEngine only supports softmax compute in FP32."
            if self.use_ixte:
                extra_transformer_engine_kwargs["use_alibi"] = args.position_embedding_type == "alibi"
                if self.ixte_v_0_2_3:
                    extra_transformer_engine_kwargs["qkv_bias"] = args.add_qkv_bias
                elif args.add_qkv_bias and not args.add_bias_linear:
                    raise NotImplementedError("Please update ixTE version to 0.2.3 to support individual qkv_bias!")
            assert (
                (bool(int(os.getenv("NVTE_APPLY_QK_LAYER_SCALING", "0"))) and args.fp16) == config.apply_query_key_layer_scaling
            ), ("Unsupported config for apply_query_key_layer_scaling in TransformerEngine. If --apply-query-key-layer-scaling is "
                "provided, set env-var NVTE_APPLY_QK_LAYER_SCALING=1 and you must be using fp16.")
            layer_tmp = transformer_engine.pytorch.TransformerLayer(
                config.hidden_size,
                config.ffn_hidden_size,
                config.num_attention_heads,
                num_gqa_groups=config.num_query_groups,
                layernorm_epsilon=config.layernorm_epsilon,
                hidden_dropout=config.hidden_dropout,
                attention_dropout=config.attention_dropout,
                init_method=config.init_method,
                output_layer_init_method=config.output_layer_init_method,
                layer_number=layer_number,
                kv_channels=config.kv_channels,
                self_attn_mask_type=self_attn_mask_type.name,
                tp_group=mpu.get_tensor_model_parallel_group() if mpu.is_initialized() else None,
                tp_size=mpu.get_tensor_model_parallel_world_size(),
                get_rng_state_tracker=get_cuda_rng_tracker
                if hasattr(get_cuda_rng_tracker(), 'is_initialized') and get_cuda_rng_tracker().is_initialized()
                else None,
                fuse_wgrad_accumulation=config.gradient_accumulation_fusion,
                seq_length=args.seq_length,
                micro_batch_size=args.micro_batch_size,
                sequence_parallel=config.sequence_parallel,
                params_dtype=config.params_dtype,
                apply_residual_connection_post_layernorm=config.apply_residual_connection_post_layernorm,
                output_layernorm=False,
                layer_type="encoder",
                drop_path_rate=self.drop_path_rates[layer_number - 1],
                set_parallel_mode=True,
                fuse_qkv_params=True,
                **extra_transformer_engine_kwargs)
            return layer_tmp

    if config.virtual_pipeline_model_parallel_size is not None:
        if args.num_layers_per_stage is None:
            assert config.num_layers % config.virtual_pipeline_model_parallel_size == 0, \
                'num_layers_per_stage must be divisible by ' \
                'virtual_pipeline_model_parallel_size'
            assert args.model_type != ModelType.encoder_and_decoder
            # Number of layers in each model chunk is the number of layers in the stage,
            # divided by the number of model chunks in a stage.
            self.num_layers = self.num_layers // config.virtual_pipeline_model_parallel_size
            # With 8 layers, 2 stages, and 4 model chunks, we want an assignment of
            # layers to stages like (each list is a model chunk):
            # Stage 0: [0]  [2]  [4]  [6]
            # Stage 1: [1]  [3]  [5]  [7]
            # With 8 layers, 2 stages, and 2 virtual stages, we want an assignment of
            # layers to stages like (each list is a model chunk):
            # Stage 0: [0, 1]  [4, 5]
            # Stage 1: [2, 3]  [6, 7]
            offset = mpu.get_virtual_pipeline_model_parallel_rank() * (
                config.num_layers // config.virtual_pipeline_model_parallel_size) + \
                (mpu.get_pipeline_model_parallel_rank() * self.num_layers)
        else:
            offset_list = [0] * len(args.num_layers_per_stage)
            for i in range(len(args.num_layers_per_stage)):
                for j in range(i):
                    offset_list[i] += args.num_layers_per_stage[j]
            offset = offset_list[mpu.get_virtual_pipeline_model_parallel_rank() * mpu.get_pipeline_model_parallel_world_size() + mpu.get_pipeline_model_parallel_rank()]
    else:
        # Each stage gets a contiguous set of layers.
        if args.model_type == ModelType.encoder_and_decoder and \
                mpu.get_pipeline_model_parallel_world_size() > 1:
            pipeline_rank = mpu.get_pipeline_model_parallel_rank()
            if layer_type == LayerType.encoder:
                offset = pipeline_rank * self.num_layers
            else:
                num_ranks_in_enc = args.pipeline_model_parallel_split_rank
                offset = (pipeline_rank - num_ranks_in_enc) * self.num_layers
        else:
            if args.num_layers_per_stage is not None:
                offset_list = [0] * len(args.num_layers_per_stage)
                for i in range(len(args.num_layers_per_stage)):
                    for j in range(i):
                        offset_list[i] += args.num_layers_per_stage[j]
                offset = offset_list[mpu.get_pipeline_model_parallel_rank()]
            else:
                offset = mpu.get_pipeline_model_parallel_rank() * self.num_layers

    if self.num_layers == 0:
        # When a standalone embedding stage is used (e.g.,
        # args.standalone_embedding_stage == True), virtual pipeline ranks
        # on pipeline rank 0 will have zero transformer layers assigned to
        # them. This results in the model's input and output tensors to be
        # the same, which will cause failure for certain output tensor
        # optimizations (e.g., pipeline output deallocation). To remedy
        # this, we assign a 'no-op' layer on these ranks, which will
        # disconnect the input tensor from the output tensor.
        self.num_layers = 1
        self.layers = torch.nn.ModuleList([ NoopTransformerLayer(1) ])
    else:
        # Build the layers
        if not args.deepspeed:
            self.layers = torch.nn.ModuleList(
                [build_layer(i + 1 + offset) for i in range(self.num_layers)])
        else:
            self.layers = []
            num_experts = args.ds_num_experts
            experts_per_layer = get_num_experts_per_layer(num_experts, self.num_layers, args.expert_interval, offset)
            for i in range(self.num_layers):
                layer_num = i + 1 + offset
                n_e = experts_per_layer[i]
                self.layers.append(build_layer(layer_num, n_e))
            self.layers = torch.nn.ModuleList(self.layers)

        # Update dropout rate for Retro encoder.
        if model_type == ModelType.retro_encoder:
            for layer in self.layers:
                if layer.self_attention.use_flash_attn:
                    layer.self_attention.core_attention_flash.dropout_p = \
                        torch.nn.Dropout(args.retro_encoder_attention_dropout)
                else:
                    layer.self_attention.core_attention.attention_dropout.p =\
                        args.retro_encoder_attention_dropout
                layer.hidden_dropout = args.retro_encoder_hidden_dropout

    if self.post_process and self.post_norm:
        # Final layer norm before output.
        self.final_norm = get_norm(config)

def parallel_transformer__checkpointed_forward_wrapper(fn):
    @wraps(fn)
    def wrapper(
        self, hidden_states, attention_mask,
        encoder_output, enc_dec_attn_mask,
        rotary_pos_emb, is_first_microbatch):
        args = get_args()

        if not args.deepspeed:
            return fn(self, hidden_states, attention_mask,
                      encoder_output, enc_dec_attn_mask,
                      rotary_pos_emb, is_first_microbatch)

        """Forward method with activation checkpointing."""
        def custom(start, end):
            def custom_forward(*args, **kwargs):
                x_, *args = args
                moe_losses = []
                for index in range(start, end):
                    # Is recompute last layer
                    # Network last layer also can be optimized, because vocab gemm always save forward tenor for backward!
                    layer = self._get_layer(index)
                    is_recompute_lastlayer = kwargs.pop('is_recompute_forward', False) and index == end - 1
                    can_opt_last_gemm = hasattr(layer, 'mlp') and hasattr(layer.mlp, 'fc2') and hasattr(layer.mlp.fc2, 'is_recompute_lastlayer')
                    if can_opt_last_gemm:
                        layer.mlp.fc2.is_recompute_lastlayer = is_recompute_lastlayer
                    output = layer(x_, *args, **kwargs)
                    if can_opt_last_gemm:
                        layer.mlp.fc2.is_recompute_lastlayer = False
                    if isinstance(output, tuple):
                        x_, moe_loss = output
                    else:
                        x_ = output
                        moe_loss = torch.tensor(0.0, device=x_.device, dtype=x_.dtype, requires_grad=True)
                    moe_losses.append(moe_loss)
                return (x_, *moe_losses)
            return custom_forward
        
        if args.deepspeed and args.deepspeed_activation_checkpointing:
            moe_losses = []
            # Make sure memory is freed.
            tensor_parallel.reset_checkpointed_activations_memory_buffer()
            l = 0
            while l < self.num_layers:
                hidden_states, *local_moe_losses = tensor_parallel.checkpoint(
                    custom(l, l + self.checkpoint_num_layers), False,
                    hidden_states, attention_mask, encoder_output, enc_dec_attn_mask,
                    None, None, None, None, rotary_pos_emb)
                moe_losses.extend(local_moe_losses)
                l += self.checkpoint_num_layers

            return hidden_states, moe_losses
        else:
            moe_losses = []
            te_forward_kwargs = {}
            if self.transformer_impl == 'transformer_engine':
                te_forward_kwargs['is_first_microbatch'] = is_first_microbatch
                if self.transformer_engine_v_0_10:
                    te_forward_kwargs['rotary_pos_emb'] = rotary_pos_emb

            if self.recompute_method == 'uniform':
                # Uniformly divide the total number of Transformer layers and
                # checkpoint the input activation of each divided chunk.
                # A method to further reduce memory usage reducing checkpoints.
                l = 0
                while l < self.num_layers:
                    if self.transformer_impl == 'transformer_engine':
                        hidden_states, *local_moe_losses = transformer_engine.pytorch.checkpoint(
                            custom(l, l + self.recompute_num_layers),
                            self.distribute_saved_activations,
                            tensor_parallel.get_cuda_rng_tracker,
                            mpu.get_tensor_model_parallel_group(),
                            hidden_states, attention_mask, encoder_output,
                            enc_dec_attn_mask, **te_forward_kwargs)
                    else:
                        hidden_states, *local_moe_losses = tensor_parallel.checkpoint(
                            custom(l, l + self.recompute_num_layers),
                            self.distribute_saved_activations,
                            hidden_states, attention_mask,
                            encoder_output, enc_dec_attn_mask,
                            None, None, None, None, rotary_pos_emb)
                    moe_losses.extend(local_moe_losses)
                    l += self.recompute_num_layers
            elif self.recompute_method == 'block':
                # Checkpoint the input activation of only a set number of individual
                # Transformer layers and skip the rest.
                # A method fully use the device memory removing redundant re-computation.
                for l in range(self.num_layers):
                    if l < self.recompute_num_layers:
                        if self.transformer_impl == 'transformer_engine':
                            hidden_states, *local_moe_losses = transformer_engine.pytorch.checkpoint(
                                custom(l, l + 1),
                                self.distribute_saved_activations,
                                tensor_parallel.get_cuda_rng_tracker,
                                mpu.get_tensor_model_parallel_group(),
                                hidden_states, attention_mask, encoder_output,
                                enc_dec_attn_mask, **te_forward_kwargs)
                        else:
                            hidden_states, *local_moe_losses = tensor_parallel.checkpoint(
                                custom(l, l + 1),
                                self.distribute_saved_activations,
                                hidden_states, attention_mask,
                                encoder_output, enc_dec_attn_mask,
                                None, None, None, None, rotary_pos_emb)
                    else:
                        if self.transformer_impl == 'transformer_engine':
                            hidden_states, *local_moe_losses = custom(l, l + 1)(
                                hidden_states, attention_mask, encoder_output,
                                enc_dec_attn_mask, **te_forward_kwargs)
                        else:
                            hidden_states, *local_moe_losses = custom(l, l + 1)(
                                hidden_states, attention_mask,
                                encoder_output, enc_dec_attn_mask,
                                None, None, None, None, rotary_pos_emb)
                            
                    moe_losses.extend(local_moe_losses)
            else:
                raise ValueError("Invalid activation recompute method.")
            return hidden_states, moe_losses
    return wrapper

def parallel_transformer_forward_wrapper(fn):
    @wraps(fn)
    def wrapper(
        self, hidden_states, attention_mask,
        encoder_output=None, enc_dec_attn_mask=None,
        retriever_input=None,
        retriever_output=None,
        retriever_attn_mask=None,
        inference_params=None,
        rotary_pos_emb=None,
        position_ids=None):
        # hidden_states: [s, b, h]

        args = get_args()
        # if not args.deepspeed:
        #     return fn(self, hidden_states, attention_mask,
        #             encoder_output=encoder_output, enc_dec_attn_mask=enc_dec_attn_mask,
        #             retriever_input=retriever_input,
        #             retriever_output=retriever_output,
        #             retriever_attn_mask=retriever_attn_mask,
        #             inference_params=inference_params,
        #             rotary_pos_emb=rotary_pos_emb)

        # Checks.
        if inference_params:
            assert self.recompute_granularity is None, \
                'inference does not work with activation checkpointing'

        if not self.pre_process:
            # See set_input_tensor()
            hidden_states = self.input_tensor

        # Viewless tensor.
        # - We only need to create a viewless tensor in the case of micro batch
        #   size (mbs) == 1, since in this case, 'hidden_states.transpose()'
        #   above creates a view tensor, and '.contiguous()' is a pass-through.
        #   For mbs >= 2, '.contiguous()' creates a new tensor, eliminating
        #   the need to make it viewless.
        #
        #   However, we don't explicitly check mbs == 1 here because
        #   make_viewless_tensor() has negligible overhead when its input
        #   is already viewless.
        #
        # - For the 'else' case above, calling make_viewless_tensor() here is
        #   likely redundant, since p2p_communication.py (likely originator)
        #   already creates viewless tensors. That said, make_viewless_tensor()
        #   is called here to be future-proof and corner-case-proof.
        hidden_states = core.utils.make_viewless_tensor(
            hidden_states,
            requires_grad=True,
            keep_graph=True,
        )

        # RNG context.
        if self.sequence_parallel and not inference_params:
            rng_context = tensor_parallel.get_cuda_rng_tracker().fork()
        else:
            rng_context = nullcontext()

        # Forward layers.
        with rng_context:
            # The fp8_autocast context manager is a no-op when enabled=True
            # The if...else serves to short circuit name resolution for fp8_autocast
            with transformer_engine.pytorch.fp8_autocast(
                enabled=self.use_fp8,
                fp8_recipe=self.fp8_recipe,
                fp8_group=self.fp8_group
            ) if self.use_fp8 else nullcontext():
                # Determine if the current iteration is first microbatch
                if self.num_microbatches_in_previous_step != get_num_microbatches():
                    self.microbatch_count = 0 # Reset count on new batch size rampup interval
                self.num_microbatches_in_previous_step = get_num_microbatches()
                is_first_microbatch = self.microbatch_count % get_num_microbatches() == 0

                # Forward pass.
                moe_losses = []
                if self.deepspeed and self.checkpoint_activations:
                    # deepspeed use the args.checkpoint_activations
                    hidden_states, moe_losses = self._checkpointed_forward(hidden_states,
                                                                attention_mask,
                                                                encoder_output,
                                                                enc_dec_attn_mask,
                                                                rotary_pos_emb,
                                                                is_first_microbatch)
                elif self.recompute_granularity == 'full':
                    # megatron-lm use args.recompute_granularity and args.recompute_method
                    hidden_states = self._checkpointed_forward(hidden_states,
                                                                attention_mask,
                                                                encoder_output,
                                                                enc_dec_attn_mask,
                                                                rotary_pos_emb,
                                                                is_first_microbatch)
                else:
                    forward_kwargs = {
                        'encoder_output': encoder_output,
                        'enc_dec_attn_mask': enc_dec_attn_mask,
                        'inference_params': inference_params,
                    }

                    if self.transformer_impl == 'transformer_engine':
                        forward_kwargs['is_first_microbatch'] = is_first_microbatch
                        forward_kwargs['checkpoint_core_attention'] = self.checkpoint_core_attention
                        if self.transformer_engine_v_0_10:
                            forward_kwargs['rotary_pos_emb'] = rotary_pos_emb
                    else:
                        forward_kwargs['rotary_pos_emb'] = rotary_pos_emb
                        forward_kwargs['retriever_input'] = retriever_input
                        forward_kwargs['retriever_output'] = retriever_output
                        forward_kwargs['retriever_attn_mask'] = retriever_attn_mask
                        forward_kwargs['position_ids'] = position_ids

                    for index in range(self.num_layers):
                        layer = self._get_layer(index)

                        hidden_states = layer(
                            hidden_states,
                            attention_mask,
                            **forward_kwargs)

                        # First Retro decoder layer returns both hidden_states
                        # and retriever_output. Make retriever_output available
                        # to subsequence Retro layers.
                        if isinstance(hidden_states, tuple):
                            assert (len(hidden_states) == 2 or len(hidden_states) == 3)
                            if len(hidden_states) == 2:
                                if not self.ds_inference:
                                    hidden_states, moe_loss = hidden_states
                                    moe_losses.append(moe_loss)
                            else:
                                forward_kwargs["retriever_output"] = hidden_states[1]
                                if not self.ds_inference:
                                    hidden_states, _, moe_loss = hidden_states
                                    moe_losses.append(moe_loss)

                # Skip counter update for eval and activation checkpointing
                if torch.is_grad_enabled() and self.training:
                    self.microbatch_count += 1

        # Final layer norm.
        if self.post_process and self.post_norm:
            hidden_states = self.final_norm(hidden_states)

        if args.deepspeed:
            return (hidden_states, *moe_losses)
        else:
            return hidden_states
    return wrapper

# class ParallelTransformer(MegatronModule):
#     """Transformer class."""

#     def __init__(self, config,
#                  model_type, layer_type=LayerType.encoder,
#                  self_attn_mask_type=AttnMaskType.padding,
#                  post_norm=True,
#                  pre_process=True,
#                  post_process=True,
#                  drop_path_rate=0.0):
#         super(ParallelTransformer, self).__init__()
#         args = get_args()

#         self.layer_type = layer_type
#         self.model_type = model_type
#         self.bf16 = config.bf16
#         self.fp32_residual_connection = config.fp32_residual_connection
#         self.post_norm = post_norm
#         self.pre_process = pre_process
#         self.post_process = post_process
#         self.input_tensor = None
#         self.drop_path_rate = drop_path_rate
#         self.transformer_impl = args.transformer_impl
#         self.retro_add_retriever = args.retro_add_retriever
#         self.ds_inference = args.ds_inference
#         self.deepspeed = args.deepspeed

#         # Store activation checkpoiting flag.
#         self.checkpoint_activations = args.checkpoint_activations
#         self.checkpoint_num_layers = args.checkpoint_num_layers
#         self.recompute_granularity = config.recompute_granularity
#         if args.recompute_method_per_stage != None:
#             if args.virtual_pipeline_model_parallel_size != None:
#                 if args.recompute_method_per_stage[mpu.get_virtual_pipeline_model_parallel_rank() * args.pipeline_model_parallel_size + mpu.get_pipeline_model_parallel_rank()] == 0:
#                     self.recompute_method = 'uniform'
#                 elif args.recompute_method_per_stage[mpu.get_virtual_pipeline_model_parallel_rank() * args.pipeline_model_parallel_size + mpu.get_pipeline_model_parallel_rank()] == 1:
#                     self.recompute_method = 'block'
#             else:
#                 if args.recompute_method_per_stage[mpu.get_pipeline_model_parallel_rank()] == 0:
#                     self.recompute_method = 'uniform'
#                 elif args.recompute_method_per_stage[mpu.get_pipeline_model_parallel_rank()] == 1:
#                     self.recompute_method = 'block'
#         else:
#             self.recompute_method = config.recompute_method
#         if args.recompute_num_layers_per_stage != None:
#             if args.virtual_pipeline_model_parallel_size != None:
#                 self.recompute_num_layers = args.recompute_num_layers_per_stage[mpu.get_virtual_pipeline_model_parallel_rank() * args.pipeline_model_parallel_size + mpu.get_pipeline_model_parallel_rank()]
#             else:
#                 self.recompute_num_layers = args.recompute_num_layers_per_stage[mpu.get_pipeline_model_parallel_rank()]
#         else:
#             self.recompute_num_layers = config.recompute_num_layers
#         self.distribute_saved_activations = \
#             config.distribute_saved_activations and not config.sequence_parallel

#         self.sequence_parallel = config.sequence_parallel

#         # Transformer Engine Init.
#         self.transformer_engine_rope_available = False
#         self.transformer_engine_v_0_10 = False
#         self.transformer_engine_v_0_11 = False
#         self.transformer_engine_v_0_8 = False
#         self.use_ixte = False
#         if self.transformer_impl == 'transformer_engine':
#             global transformer_engine
#             import transformer_engine
#             from importlib.metadata import version
#             from pkg_resources import packaging

#             if ixte_extensions._USE_IXTE:
#                 te_version = packaging.version.Version(ixte_extensions.te_version())
#                 self.use_ixte = True
#             else:
#                 te_version = packaging.version.Version(version("transformer-engine"))
#             if te_version >= packaging.version.Version("0.8.0"):
#                 self.transformer_engine_v_0_8 = True
#             if te_version >= packaging.version.Version("0.10.0"):
#                 self.transformer_engine_v_0_10 = True
#             if te_version >= packaging.version.Version("0.11.0"):
#                 self.transformer_engine_v_0_11 = True
#             if te_version >= packaging.version.Version("0.10.0"):
#                 self.transformer_engine_rope_available = True

#             del version, packaging

#             assert not args.squared_relu, "TransformerEngine does not support squared relu activation."

#         self.use_fp8 = args.fp8 is not None
#         self.fp8_recipe = None
#         self.fp8_group = None
#         if self.use_fp8:
#             assert args.transformer_impl == 'transformer_engine', \
#                 'transformer-engine required for fp8 training and inference'
#             self.fp8_group = mpu.get_amax_reduction_group()
#             if args.fp8 == "e4m3":
#                 fp8_format = transformer_engine.common.recipe.Format.E4M3
#             elif args.fp8 == "hybrid":
#                 fp8_format = transformer_engine.common.recipe.Format.HYBRID
#             else:
#                 raise ValueError("The DelayedScaling recipe only supports E4M3 and HYBRID formats.")
#             self.fp8_recipe = transformer_engine.common.recipe.DelayedScaling(
#                 margin=args.fp8_margin,
#                 interval=args.fp8_interval,
#                 fp8_format=fp8_format,
#                 amax_history_len=args.fp8_amax_history_len,
#                 amax_compute_algo=args.fp8_amax_compute_algo,
#                 override_linear_precision=(False, False, not args.fp8_wgrad),
#             )

#         self.num_microbatches_in_previous_step = -1
#         self.microbatch_count = 0
#         self.checkpoint_core_attention = config.recompute_granularity == 'selective'

#         # Number of layers.
#         self.num_layers = _get_num_layers(args, model_type,
#                                           layer_type==LayerType.decoder)

#         self.drop_path_rates = [
#             rate.item() for rate in
#             torch.linspace(0, self.drop_path_rate, config.num_layers)]

#         self.retro_layer_numbers = None
#         if model_type == ModelType.retro_decoder:
#             retro_layer_start = 6 if config.num_layers <= 15 else 9
#             self.retro_layer_numbers = \
#                 np.arange(retro_layer_start, args.num_layers + 1, 3).tolist()
#         if model_type == ModelType.retro_encoder:
#             self.retro_layer_numbers = [1]

#         # Transformer layers.
#         if args.retro_add_retriever:
#             assert self.recompute_granularity != 'full', \
#                 "Full recompute not supported for Retro."
#             assert args.transformer_impl == 'local', \
#                 "Transformer engine does not support Retro layers."
#         def build_layer(layer_number, n_e=1):
#             if args.transformer_impl == 'local':
#                 current_layer_type = _get_layer_type(
#                     model_type, layer_type, self.retro_layer_numbers,
#                     layer_number)
#                 return ParallelTransformerLayer(
#                     config,
#                     layer_number,
#                     layer_type=current_layer_type,
#                     self_attn_mask_type=self_attn_mask_type,
#                     drop_path_rate=self.drop_path_rates[layer_number - 1],
#                     num_experts=n_e)
#             else:
#                 # This argument is only available from TE v0.10 onwards.
#                 extra_transformer_engine_kwargs = {}
#                 if self.transformer_engine_v_0_8:
#                     extra_transformer_engine_kwargs["bias"] = args.add_bias_linear
#                 if self.transformer_engine_v_0_10:
#                     extra_transformer_engine_kwargs["activation"] = "swiglu" if args.swiglu else "gelu"
#                 if self.transformer_engine_v_0_11:
#                     extra_transformer_engine_kwargs["normalization"] = args.normalization
#                 if not ixte_extensions._USE_IXTE:
#                     assert config.attention_softmax_in_fp32, "TransformerEngine only supports softmax compute in FP32."
#                 if self.use_ixte:
#                     extra_transformer_engine_kwargs["use_alibi"] = args.position_embedding_type == "alibi"
#                 assert (
#                     (bool(int(os.getenv("NVTE_APPLY_QK_LAYER_SCALING", "0"))) and args.fp16) == config.apply_query_key_layer_scaling
#                 ), ("Unsupported config for apply_query_key_layer_scaling in TransformerEngine. If --apply-query-key-layer-scaling is "
#                     "provided, set env-var NVTE_APPLY_QK_LAYER_SCALING=1 and you must be using fp16.")
#                 layer_tmp = transformer_engine.pytorch.TransformerLayer(
#                     config.hidden_size,
#                     config.ffn_hidden_size,
#                     config.num_attention_heads,
#                     num_gqa_groups=config.num_query_groups,
#                     layernorm_epsilon=config.layernorm_epsilon,
#                     hidden_dropout=config.hidden_dropout,
#                     attention_dropout=config.attention_dropout,
#                     init_method=config.init_method,
#                     output_layer_init_method=config.output_layer_init_method,
#                     layer_number=layer_number,
#                     kv_channels=config.kv_channels,
#                     self_attn_mask_type=self_attn_mask_type.name,
#                     tp_group=mpu.get_tensor_model_parallel_group() if mpu.is_initialized() else None,
#                     tp_size=mpu.get_tensor_model_parallel_world_size(),
#                     get_rng_state_tracker=get_cuda_rng_tracker
#                     if get_cuda_rng_tracker().is_initialized()
#                     else None,
#                     fuse_wgrad_accumulation=config.gradient_accumulation_fusion,
#                     seq_length=args.seq_length,
#                     micro_batch_size=args.micro_batch_size,
#                     sequence_parallel=config.sequence_parallel,
#                     params_dtype=config.params_dtype,
#                     apply_residual_connection_post_layernorm=config.apply_residual_connection_post_layernorm,
#                     output_layernorm=False,
#                     layer_type="encoder",
#                     drop_path_rate=self.drop_path_rates[layer_number - 1],
#                     set_parallel_mode=True,
#                     fuse_qkv_params=True,
#                     **extra_transformer_engine_kwargs)
#                 return layer_tmp

#         if config.virtual_pipeline_model_parallel_size is not None:
#             if args.num_layers_per_stage is None:
#                 assert config.num_layers % config.virtual_pipeline_model_parallel_size == 0, \
#                     'num_layers_per_stage must be divisible by ' \
#                     'virtual_pipeline_model_parallel_size'
#                 assert args.model_type != ModelType.encoder_and_decoder
#                 # Number of layers in each model chunk is the number of layers in the stage,
#                 # divided by the number of model chunks in a stage.
#                 self.num_layers = self.num_layers // config.virtual_pipeline_model_parallel_size
#                 # With 8 layers, 2 stages, and 4 model chunks, we want an assignment of
#                 # layers to stages like (each list is a model chunk):
#                 # Stage 0: [0]  [2]  [4]  [6]
#                 # Stage 1: [1]  [3]  [5]  [7]
#                 # With 8 layers, 2 stages, and 2 virtual stages, we want an assignment of
#                 # layers to stages like (each list is a model chunk):
#                 # Stage 0: [0, 1]  [4, 5]
#                 # Stage 1: [2, 3]  [6, 7]
#                 offset = mpu.get_virtual_pipeline_model_parallel_rank() * (
#                     config.num_layers // config.virtual_pipeline_model_parallel_size) + \
#                     (mpu.get_pipeline_model_parallel_rank() * self.num_layers)
#             else:
#                 offset_list = [0] * len(args.num_layers_per_stage)
#                 for i in range(len(args.num_layers_per_stage)):
#                     for j in range(i):
#                         offset_list[i] += args.num_layers_per_stage[j]
#                 offset = offset_list[mpu.get_virtual_pipeline_model_parallel_rank() * mpu.get_pipeline_model_parallel_world_size() + mpu.get_pipeline_model_parallel_rank()]
#         else:
#             # Each stage gets a contiguous set of layers.
#             if args.model_type == ModelType.encoder_and_decoder and \
#                     mpu.get_pipeline_model_parallel_world_size() > 1:
#                 pipeline_rank = mpu.get_pipeline_model_parallel_rank()
#                 if layer_type == LayerType.encoder:
#                     offset = pipeline_rank * self.num_layers
#                 else:
#                     num_ranks_in_enc = args.pipeline_model_parallel_split_rank
#                     offset = (pipeline_rank - num_ranks_in_enc) * self.num_layers
#             else:
#                 if args.num_layers_per_stage is not None:
#                     offset_list = [0] * len(args.num_layers_per_stage)
#                     for i in range(len(args.num_layers_per_stage)):
#                         for j in range(i):
#                             offset_list[i] += args.num_layers_per_stage[j]
#                     offset = offset_list[mpu.get_pipeline_model_parallel_rank()]
#                 else:
#                     offset = mpu.get_pipeline_model_parallel_rank() * self.num_layers

#         if self.num_layers == 0:
#             # When a standalone embedding stage is used (e.g.,
#             # args.standalone_embedding_stage == True), virtual pipeline ranks
#             # on pipeline rank 0 will have zero transformer layers assigned to
#             # them. This results in the model's input and output tensors to be
#             # the same, which will cause failure for certain output tensor
#             # optimizations (e.g., pipeline output deallocation). To remedy
#             # this, we assign a 'no-op' layer on these ranks, which will
#             # disconnect the input tensor from the output tensor.
#             self.num_layers = 1
#             self.layers = torch.nn.ModuleList([ NoopTransformerLayer(1) ])
#         else:
#             # Build the layers
#             if not args.deepspeed:
#                 self.layers = torch.nn.ModuleList(
#                     [build_layer(i + 1 + offset) for i in range(self.num_layers)])
#             else:
#                 self.layers = []
#                 num_experts = args.ds_num_experts
#                 experts_per_layer = get_num_experts_per_layer(num_experts, self.num_layers, args.expert_interval, offset)
#                 for i in range(self.num_layers):
#                     layer_num = i + 1 + offset
#                     n_e = experts_per_layer[i]
#                     self.layers.append(build_layer(layer_num, n_e))
#                 self.layers = torch.nn.ModuleList(self.layers)

#             # Update dropout rate for Retro encoder.
#             if model_type == ModelType.retro_encoder:
#                 for layer in self.layers:
#                     if layer.self_attention.use_flash_attn:
#                         layer.self_attention.core_attention_flash.dropout_p = \
#                             torch.nn.Dropout(args.retro_encoder_attention_dropout)
#                     else:
#                         layer.self_attention.core_attention.attention_dropout.p =\
#                             args.retro_encoder_attention_dropout
#                     layer.hidden_dropout = args.retro_encoder_hidden_dropout

#         if self.post_process and self.post_norm:
#             # Final layer norm before output.
#             self.final_norm = get_norm(config)

#     def _get_layer(self, layer_number):
#         return self.layers[layer_number]

#     def _checkpointed_forward(self, hidden_states, attention_mask,
#                               encoder_output, enc_dec_attn_mask,
#                               rotary_pos_emb, is_first_microbatch):
#         args = get_args()

#         """Forward method with activation checkpointing."""
#         def custom(start, end):
#             def custom_forward(*args, **kwargs):
#                 x_, *args = args
#                 moe_losses = []
#                 for index in range(start, end):
#                     # Is recompute last layer
#                     # Network last layer also can be optimized, because vocab gemm always save forward tenor for backward!
#                     if self.transformer_impl == 'transformer_engine' and ixte_extensions._USE_IXTE:
#                         kwargs["is_recompute_lastlayer"] = index == end - 1
#                     layer = self._get_layer(index)
#                     output = layer(x_, *args, **kwargs)
#                     if isinstance(output, tuple):
#                         x_, moe_loss = output
#                     else:
#                         x_ = output
#                         moe_loss = torch.tensor(0.0, device=x_.device, dtype=x_.dtype, requires_grad=True)
#                     moe_losses.append(moe_loss)
#                 return (x_, *moe_losses)
#             return custom_forward
        
#         if args.deepspeed and args.deepspeed_activation_checkpointing:
#             moe_losses = []
#             # Make sure memory is freed.
#             tensor_parallel.reset_checkpointed_activations_memory_buffer()
#             l = 0
#             while l < self.num_layers:
#                 hidden_states, *local_moe_losses = tensor_parallel.checkpoint(
#                     custom(l, l + self.checkpoint_num_layers), False,
#                     hidden_states, attention_mask, encoder_output, enc_dec_attn_mask,
#                     None, None, None, None, rotary_pos_emb)
#                 moe_losses.extend(local_moe_losses)
#                 l += self.checkpoint_num_layers

#             return hidden_states, moe_losses
#         else:
#             moe_losses = []
#             te_forward_kwargs = {}
#             if self.transformer_impl == 'transformer_engine':
#                 te_forward_kwargs['is_first_microbatch'] = is_first_microbatch
#                 if self.transformer_engine_v_0_10:
#                     te_forward_kwargs['rotary_pos_emb'] = rotary_pos_emb

#             if self.recompute_method == 'uniform':
#                 # Uniformly divide the total number of Transformer layers and
#                 # checkpoint the input activation of each divided chunk.
#                 # A method to further reduce memory usage reducing checkpoints.
#                 l = 0
#                 while l < self.num_layers:
#                     if self.transformer_impl == 'transformer_engine':
#                         hidden_states, *local_moe_losses = transformer_engine.pytorch.checkpoint(
#                             custom(l, l + self.recompute_num_layers),
#                             self.distribute_saved_activations,
#                             tensor_parallel.get_cuda_rng_tracker,
#                             mpu.get_tensor_model_parallel_group(),
#                             hidden_states, attention_mask, encoder_output,
#                             enc_dec_attn_mask, **te_forward_kwargs)
#                     else:
#                         hidden_states, *local_moe_losses = tensor_parallel.checkpoint(
#                             custom(l, l + self.recompute_num_layers),
#                             self.distribute_saved_activations,
#                             hidden_states, attention_mask,
#                             encoder_output, enc_dec_attn_mask,
#                             None, None, None, None, rotary_pos_emb)
#                     moe_losses.extend(local_moe_losses)
#                     l += self.recompute_num_layers
#             elif self.recompute_method == 'block':
#                 # Checkpoint the input activation of only a set number of individual
#                 # Transformer layers and skip the rest.
#                 # A method fully use the device memory removing redundant re-computation.
#                 for l in range(self.num_layers):
#                     if l < self.recompute_num_layers:
#                         if self.transformer_impl == 'transformer_engine':
#                             hidden_states, *local_moe_losses = transformer_engine.pytorch.checkpoint(
#                                 custom(l, l + 1),
#                                 self.distribute_saved_activations,
#                                 tensor_parallel.get_cuda_rng_tracker,
#                                 mpu.get_tensor_model_parallel_group(),
#                                 hidden_states, attention_mask, encoder_output,
#                                 enc_dec_attn_mask, **te_forward_kwargs)
#                         else:
#                             hidden_states, *local_moe_losses = tensor_parallel.checkpoint(
#                                 custom(l, l + 1),
#                                 self.distribute_saved_activations,
#                                 hidden_states, attention_mask,
#                                 encoder_output, enc_dec_attn_mask,
#                                 None, None, None, None, rotary_pos_emb)
#                     else:
#                         if self.transformer_impl == 'transformer_engine':
#                             hidden_states, *local_moe_losses = custom(l, l + 1)(
#                                 hidden_states, attention_mask, encoder_output,
#                                 enc_dec_attn_mask, **te_forward_kwargs)
#                         else:
#                             hidden_states, *local_moe_losses = custom(l, l + 1)(
#                                 hidden_states, attention_mask,
#                                 encoder_output, enc_dec_attn_mask,
#                                 None, None, None, None, rotary_pos_emb)
                            
#                     moe_losses.extend(local_moe_losses)
#             else:
#                 raise ValueError("Invalid activation recompute method.")
#             return hidden_states, moe_losses

#     def set_input_tensor(self, input_tensor):
#         """Set input tensor to be used instead of forward()'s input.

#         When doing pipeline parallelism the input from the previous
#         stage comes from communication, not from the input, so the
#         model's forward_step_func won't have it. This function is thus
#         used by internal code to bypass the input provided by the
#         forward_step_func"""
#         self.input_tensor = input_tensor

#     def forward(self, hidden_states, attention_mask,
#                 encoder_output=None, enc_dec_attn_mask=None,
#                 retriever_input=None,
#                 retriever_output=None,
#                 retriever_attn_mask=None,
#                 inference_params=None,
#                 rotary_pos_emb=None):
#         # hidden_states: [s, b, h]

#         # Checks.
#         if inference_params:
#             assert self.recompute_granularity is None, \
#                 'inference does not work with activation checkpointing'

#         if not self.pre_process:
#             # See set_input_tensor()
#             hidden_states = self.input_tensor

#         # Viewless tensor.
#         # - We only need to create a viewless tensor in the case of micro batch
#         #   size (mbs) == 1, since in this case, 'hidden_states.transpose()'
#         #   above creates a view tensor, and '.contiguous()' is a pass-through.
#         #   For mbs >= 2, '.contiguous()' creates a new tensor, eliminating
#         #   the need to make it viewless.
#         #
#         #   However, we don't explicitly check mbs == 1 here because
#         #   make_viewless_tensor() has negligible overhead when its input
#         #   is already viewless.
#         #
#         # - For the 'else' case above, calling make_viewless_tensor() here is
#         #   likely redundant, since p2p_communication.py (likely originator)
#         #   already creates viewless tensors. That said, make_viewless_tensor()
#         #   is called here to be future-proof and corner-case-proof.
#         hidden_states = core.utils.make_viewless_tensor(
#             hidden_states,
#             requires_grad=True,
#             keep_graph=True,
#         )

#         # RNG context.
#         if self.sequence_parallel:
#             rng_context = tensor_parallel.get_cuda_rng_tracker().fork()
#         else:
#             rng_context = nullcontext()

#         # Forward layers.
#         with rng_context:
#             # The fp8_autocast context manager is a no-op when enabled=True
#             # The if...else serves to short circuit name resolution for fp8_autocast
#             with transformer_engine.pytorch.fp8_autocast(
#                 enabled=self.use_fp8,
#                 fp8_recipe=self.fp8_recipe,
#                 fp8_group=self.fp8_group
#             ) if self.use_fp8 else nullcontext():
#                 # Determine if the current iteration is first microbatch
#                 if self.num_microbatches_in_previous_step != get_num_microbatches():
#                     self.microbatch_count = 0 # Reset count on new batch size rampup interval
#                 self.num_microbatches_in_previous_step = get_num_microbatches()
#                 is_first_microbatch = self.microbatch_count % get_num_microbatches() == 0

#                 # Forward pass.
#                 moe_losses = []
#                 if self.deepspeed and self.checkpoint_activations:
#                     hidden_states, moe_losses = self._checkpointed_forward(hidden_states,
#                                                                attention_mask,
#                                                                encoder_output,
#                                                                enc_dec_attn_mask,
#                                                                rotary_pos_emb,
#                                                                is_first_microbatch)
#                 elif self.recompute_granularity == 'full':
#                     hidden_states, moe_losses = self._checkpointed_forward(hidden_states,
#                                                                attention_mask,
#                                                                encoder_output,
#                                                                enc_dec_attn_mask,
#                                                                rotary_pos_emb,
#                                                                is_first_microbatch)
#                 else:
#                     forward_kwargs = {
#                         'encoder_output': encoder_output,
#                         'enc_dec_attn_mask': enc_dec_attn_mask,
#                         'inference_params': inference_params,
#                     }

#                     if self.transformer_impl == 'transformer_engine':
#                         forward_kwargs['is_first_microbatch'] = is_first_microbatch
#                         forward_kwargs['checkpoint_core_attention'] = self.checkpoint_core_attention
#                         if self.transformer_engine_v_0_10:
#                             forward_kwargs['rotary_pos_emb'] = rotary_pos_emb
#                     else:
#                         forward_kwargs['rotary_pos_emb'] = rotary_pos_emb
#                         forward_kwargs['retriever_input'] = retriever_input
#                         forward_kwargs['retriever_output'] = retriever_output
#                         forward_kwargs['retriever_attn_mask'] = retriever_attn_mask

#                     for index in range(self.num_layers):
#                         layer = self._get_layer(index)

#                         hidden_states = layer(
#                             hidden_states,
#                             attention_mask,
#                             **forward_kwargs)

#                         # First Retro decoder layer returns both hidden_states
#                         # and retriever_output. Make retriever_output available
#                         # to subsequence Retro layers.
#                         if isinstance(hidden_states, tuple):
#                             assert (len(hidden_states) == 2 or len(hidden_states) == 3)
#                             if len(hidden_states) == 2:
#                                 if not self.ds_inference:
#                                     hidden_states, moe_loss = hidden_states
#                                     moe_losses.append(moe_loss)
#                             else:
#                                 forward_kwargs["retriever_output"] = hidden_states[1]
#                                 if not self.ds_inference:
#                                     hidden_states, _, moe_loss = hidden_states
#                                     moe_losses.append(moe_loss)

#                 # Skip counter update for eval and activation checkpointing
#                 if torch.is_grad_enabled() and self.training:
#                     self.microbatch_count += 1

#         # Final layer norm.
#         if self.post_process and self.post_norm:
#             hidden_states = self.final_norm(hidden_states)

#         return (hidden_states, *moe_losses)

#     def load_state_dict(self, state_dict, strict=True):
#         """Customize load."""

#         # Handle renaming layernorm -> norm in component names
#         state_dict_ = {}
#         for key in state_dict.keys():
#             # Bypass TransformerEngine module parameters.
#             if "layernorm_qkv" in key or "layernorm_mlp" in key:
#                 state_dict_[key] = state_dict[key]
#                 continue
#             newkey = key.replace("layernorm", "norm")
#             state_dict_[newkey] = state_dict[key]

#         super().load_state_dict(state_dict_, strict)

class LMHeadPipe(MegatronModule):
    """
    Arguments:
        vocab_size: size of vocabulary.
        hidden_size: hidden size
        gather_output: wether output logits being gathered or not.
        init_method: init method for weight initialization
        config:
    """

    def __init__(self, hidden_size, vocab_size, config):
        args = get_args()
        super(LMHeadPipe, self).__init__()
        self.lm_head = tensor_parallel.ColumnParallelLinear(input_size=hidden_size,
                                                            output_size=vocab_size,
                                                            bias=False,
                                                            config=config,
                                                            init_method=config.init_method,)

    def forward(self, inputs, **kwargs):
        assert torch.is_tensor(inputs) or isinstance(inputs, tuple)
        if isinstance(inputs, tuple):
            hidden_states = inputs[0]
        else:
            hidden_states = inputs

        if not hasattr(self, '_args'):
            self._args = get_args()

        if hasattr(self._args, 'attn_mask'):
            attention_mask = None
        else:
            attention_mask = inputs[1]

        logits, _ = self.lm_head(hidden_states)

        # If cmd args has attn_mask, we don't forward it as an activation.
        if hasattr(self._args, 'attn_mask'):
            return logits
        else:
            return logits, attention_mask
