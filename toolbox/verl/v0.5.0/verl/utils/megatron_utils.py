# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
"""Pretrain utilities."""

import gc
import os
import warnings
from typing import Any

import torch
import torch.nn.functional as F
from megatron.core import ModelParallelConfig, mpu, tensor_parallel
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.enums import ModelType
from megatron.core.optimizer import ChainedOptimizer, OptimizerConfig
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.module import Float16Module
from megatron.core.utils import get_attr_wrapped_model
from transformers import PretrainedConfig

import verl.utils.megatron.tensor_parallel as tp_utils
from verl.utils.device import get_device_id, get_device_name, get_torch_device
from verl.utils.fs import local_mkdir_safe
from verl.utils.model import normalize_model_name
from verl.utils.torch_dtypes import PrecisionType


def get_model_config(model):
    return get_attr_wrapped_model(model, "config", allow_none=False)


def get_model(
    model_provider_func,
    model_type=ModelType.encoder_or_decoder,
    wrap_with_ddp=True,
    use_distributed_optimizer=True,
    transformer_config=None,
    override_ddp_config=None,
):
    """Build the model."""
    # Build model.
    if (
        mpu.get_pipeline_model_parallel_world_size() > 1
        and mpu.get_virtual_pipeline_model_parallel_world_size() is not None
    ):
        assert model_type != ModelType.encoder_and_decoder, (
            "Interleaved schedule not supported for model with both encoder and decoder"
        )
        model = []
        for i in range(mpu.get_virtual_pipeline_model_parallel_world_size()):
            mpu.set_virtual_pipeline_model_parallel_rank(i)
            # Set pre_process and post_process only after virtual rank is set.
            pre_process = mpu.is_pipeline_first_stage()
            post_process = mpu.is_pipeline_last_stage()
            this_model = model_provider_func(pre_process=pre_process, post_process=post_process)
            this_model.model_type = model_type
            model.append(this_model)
        mpu.set_virtual_pipeline_model_parallel_rank(0)
    else:
        pre_process = mpu.is_pipeline_first_stage()
        post_process = mpu.is_pipeline_last_stage()
        add_encoder = True
        add_decoder = True
        if model_type == ModelType.encoder_and_decoder:
            if mpu.get_pipeline_model_parallel_world_size() > 1:
                assert mpu.get_pipeline_model_parallel_split_rank() is not None, (
                    "Split rank needs to be specified for model with both encoder and decoder"
                )
                rank = mpu.get_pipeline_model_parallel_rank()
                split_rank = mpu.get_pipeline_model_parallel_split_rank()
                world_size = mpu.get_pipeline_model_parallel_world_size()
                pre_process = rank == 0 or rank == split_rank
                post_process = (rank == (split_rank - 1)) or (rank == (world_size - 1))
                add_encoder = mpu.is_pipeline_stage_before_split()
                add_decoder = mpu.is_pipeline_stage_after_split()
            model = model_provider_func(
                pre_process=pre_process, post_process=post_process, add_encoder=add_encoder, add_decoder=add_decoder
            )
        else:
            model = model_provider_func(pre_process=pre_process, post_process=post_process)
        model.model_type = model_type

    if not isinstance(model, list):
        model = [model]

    # Set tensor model parallel attributes if not set.
    # Only parameters that are already tensor model parallel have these
    # attributes set for them. We should make sure the default attributes
    # are set for all params so the optimizer can use them.
    for model_module in model:
        for param in model_module.parameters():
            tensor_parallel.set_defaults_if_not_set_tensor_model_parallel_attributes(param)

    # Print number of parameters.
    if mpu.get_data_parallel_rank() == 0:
        print(
            " > number of parameters on (tensor, pipeline) model parallel rank ({}, {}): {}".format(
                mpu.get_tensor_model_parallel_rank(),
                mpu.get_pipeline_model_parallel_rank(),
                sum([sum([p.nelement() for p in model_module.parameters()]) for model_module in model]),
            ),
            flush=True,
        )

    # GPU allocation.
    if transformer_config is None or (not transformer_config.use_cpu_initialization):
        for model_module in model:
            model_module.to(f"{get_device_name()}:{get_device_id()}")

    # Fp16 conversion.
    config: TransformerConfig = get_model_config(model[0])
    config.fp8 = None
    tfconfig: TransformerConfig = model[0].config
    if config.fp16 or config.bf16:  # the ModelParallelConfig in GPTModel
        model = [Float16Module(config, model_module) for model_module in model]

    if wrap_with_ddp:
        ddp_models = []
        ddp_config_dict = {
            "use_distributed_optimizer": use_distributed_optimizer,
            "grad_reduce_in_fp32": True,
            "overlap_grad_reduce": False,
        }
        if override_ddp_config is not None:
            ddp_config_dict.update(override_ddp_config)
        ddp_config = DistributedDataParallelConfig(**ddp_config_dict)
        for model_chunk_idx, model_chunk in enumerate(model):
            ddp_model = DDP(
                config=tfconfig,
                module=model_chunk,
                disable_bucketing=(model_chunk_idx > 0),
                ddp_config=ddp_config,
            )
            ddp_models.append(ddp_model)
        model = ddp_models
        # # Broadcast params from data parallel src rank to other data parallel ranks.
        # # if args.data_parallel_random_init:
        for model_module in model:
            model_module.broadcast_params()
    return model


ALL_MODULE_WRAPPER_CLASSNAMES = (DDP, Float16Module)


def unwrap_model(model, module_instances=ALL_MODULE_WRAPPER_CLASSNAMES):
    return_list = True
    if not isinstance(model, list):
        model = [model]
        return_list = False
    unwrapped_model = []
    for model_module in model:
        while isinstance(model_module, module_instances):
            model_module = model_module.module
        unwrapped_model.append(model_module)
    if not return_list:
        return unwrapped_model[0]
    return unwrapped_model


def convert_config(hf_config: PretrainedConfig, megatron_config) -> TransformerConfig:
    print(f"megatron config {megatron_config}")
    dt = PrecisionType.to_dtype(megatron_config.params_dtype)
    print(f"pipeline_dtype=megatron_config {dt}")
    qkv_bias = True if "Qwen2ForCausalLM" in hf_config.architectures else getattr(hf_config, "attention_bias", False)
    overlap_p2p_comm = (
        mpu.get_virtual_pipeline_model_parallel_world_size() is not None
        and mpu.get_virtual_pipeline_model_parallel_world_size() > 1
    )
    batch_p2p_comm = False
    transformer_config = TransformerConfig(
        num_layers=hf_config.num_hidden_layers,
        hidden_size=hf_config.hidden_size,
        num_attention_heads=hf_config.num_attention_heads,
        num_query_groups=hf_config.num_key_value_heads,
        ffn_hidden_size=hf_config.intermediate_size,
        #    max_position_embeddings=hf_config.max_position_embeddings,
        activation_func=F.silu,
        normalization="RMSNorm",
        #    rotary_percent=False, # default,
        gated_linear_unit=True,  # for llama
        use_cpu_initialization=True,
        apply_residual_connection_post_layernorm=False,  # check what's this mean
        add_bias_linear=False,
        tensor_model_parallel_size=mpu.get_tensor_model_parallel_world_size(),
        pipeline_model_parallel_size=mpu.get_pipeline_model_parallel_world_size(),
        virtual_pipeline_model_parallel_size=mpu.get_virtual_pipeline_model_parallel_world_size(),
        context_parallel_size=mpu.get_context_parallel_world_size(),
        overlap_p2p_comm=overlap_p2p_comm,
        batch_p2p_comm=batch_p2p_comm,
        pipeline_dtype=dt,
        params_dtype=dt,
        sequence_parallel=mpu.get_tensor_model_parallel_world_size() > 1,
        variable_seq_lengths=True,
        masked_softmax_fusion=True,
        moe_token_dispatcher_type="alltoall",
        attention_dropout=hf_config.attention_dropout,
        hidden_dropout=getattr(hf_config, "hidden_dropout", 0.0),
        add_qkv_bias=qkv_bias,
        bf16=dt is torch.bfloat16,
    )

    return transformer_config


def init_megatron_optim_config(optim_config: dict) -> OptimizerConfig:
    config = OptimizerConfig(
        optimizer=optim_config.get("optimizer", "adam"),
        lr=optim_config.get("lr"),
        min_lr=optim_config.get("min_lr", None),
        clip_grad=optim_config.get("clip_grad", 1.0),
        weight_decay=optim_config.get("weight_decay", 0.01),
        bf16=True,
        params_dtype=torch.bfloat16,
        use_distributed_optimizer=True,
    )
    return config


def mcore_model_parallel_config(
    sequence_parallel: bool,
    params_dtype: torch.dtype,
) -> ModelParallelConfig:
    # WARNING: Code should not reach this point. This function is deprecated and will be removed.
    # Please use hf_to_mcore_config_dense() from verl.models.mcore.config_converter instead.
    warnings.warn(
        "Code should not reach this point. This function is deprecated and will be removed. Please use "
        "hf_to_mcore_config_dense() from verl.models.mcore.config_converter instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return ModelParallelConfig(
        tensor_model_parallel_size=mpu.get_tensor_model_parallel_world_size(),
        pipeline_model_parallel_size=mpu.get_pipeline_model_parallel_world_size(),
        virtual_pipeline_model_parallel_size=mpu.get_virtual_pipeline_model_parallel_world_size(),
        context_parallel_size=mpu.get_context_parallel_world_size(),
        sequence_parallel=sequence_parallel,
        params_dtype=params_dtype,
        pipeline_dtype=params_dtype,
        bf16=True,
        fp16=False,
        timers=None,
    )


@torch.no_grad()
def offload_megatron_model_to_cpu(models):
    """
    In megatron, the model and optimizer storage are:
    - bf16 parameter data chunked in model parallel group
    - fp32 grad chunked in model parallel group
    - fp32 main_parameter chunked in model and dp group
    - fp32 optimizer state chunked in model and dp group
    """
    for model_chunk in models:
        if isinstance(model_chunk, DDP):
            model_chunk_all_buffers = [model_chunk.buffers, model_chunk.expert_parallel_buffers]
            for buffers in model_chunk_all_buffers:
                for buffer in buffers:
                    # offload parameters
                    if buffer.param_data.storage().size() > 0:
                        buffer.param_data.cpu_data = buffer.param_data.data.cpu().pin_memory()
                        buffer.param_data_size = buffer.param_data.storage().size()
                        buffer.param_data.storage().resize_(0)

                    assert buffer.param_data_size == buffer.param_data.cpu_data.storage().size()

                    if buffer.grad_data.storage().size() > 0:
                        # if the grad_data size is already zero, we assume that it is already offloaded
                        buffer.grad_data_size = buffer.grad_data.storage().size()
                        buffer.grad_data.storage().resize_(0)
        else:
            # we need this for ref module
            for _, param in model_chunk.named_parameters():
                param.data = param.data.to("cpu", non_blocking=True)
                if param.grad is not None:
                    param.grad = param.grad.to("cpu", non_blocking=True)
    gc.collect()
    get_torch_device().empty_cache()


@torch.no_grad()
def load_megatron_model_to_gpu(models, load_grad=True):
    for model_chunk in models:
        if isinstance(model_chunk, DDP):
            model_chunk_all_buffers = [model_chunk.buffers, model_chunk.expert_parallel_buffers]
            for buffers in model_chunk_all_buffers:
                for buffer in buffers:
                    # sometimes, we don't want to load grad for pure inference
                    if load_grad:
                        buffer.grad_data.storage().resize_(buffer.grad_data_size)
                        buffer.grad_data.zero_()

                    if buffer.param_data.storage().size() == 0:
                        buffer.param_data.storage().resize_(buffer.param_data_size)
                        # copy data from cpu to cuda
                        buffer.param_data.copy_(buffer.param_data.cpu_data, non_blocking=True)
        else:
            # we need this for ref module
            device_id = get_device_id()
            for _, param in model_chunk.named_parameters():
                param.data = param.data.to(device_id, non_blocking=True)
                if param.grad is not None:
                    param.grad = param.grad.to(device_id, non_blocking=True)
    gc.collect()
    get_torch_device().empty_cache()


@torch.no_grad()
def offload_megatron_copy_params(optimizers):
    """
    Offload optimizer parameters to CPU. Supports both Megatron optimizers
    and `ChainedOptimizer`, which wraps a list of underlying optimizers.

    Args:
        optimizers: The optimizer or ChainedOptimizer instance.
    """

    def _iter_opts(opt):
        if isinstance(opt, ChainedOptimizer):
            return opt.chained_optimizers
        return [opt]

    def offload_tensor_to_cpu(tensor):
        if tensor is None:
            return
        tensor.data = tensor.data.to("cpu", non_blocking=True)

    def offload_group_to_cpu(group):
        if group is None:
            return

        if isinstance(group, list):
            for param_group in group:
                if isinstance(param_group, list):
                    for param in param_group:
                        offload_tensor_to_cpu(param)
                else:
                    offload_tensor_to_cpu(param_group)
        else:
            offload_tensor_to_cpu(group)

    # Offload all parameter groups to CPU for each underlying optimizer

    for _opt in _iter_opts(optimizers):
        if hasattr(_opt, "shard_fp32_from_float16_groups"):
            offload_group_to_cpu(_opt.shard_fp32_from_float16_groups)


@torch.no_grad()
def load_megatron_copy_params(optimizers):
    """
    Load optimizer parameters back to GPU. Handles ChainedOptimizer.

    Args:
        optimizers: Optimizer or ChainedOptimizer instance.
    """

    def _iter_opts(opt):
        if isinstance(opt, ChainedOptimizer):
            return opt.chained_optimizers
        return [opt]

    def load_tensor_to_gpu(tensor):
        if tensor is None:
            return
        device_id = get_device_id()
        tensor.data = tensor.data.to(device_id, non_blocking=True)

    def load_group_to_gpu(group):
        if group is None:
            return

        if isinstance(group, list):
            for param_group in group:
                if isinstance(param_group, list):
                    for param in param_group:
                        load_tensor_to_gpu(param)
                else:
                    load_tensor_to_gpu(param_group)
        else:
            load_tensor_to_gpu(group)

    # Load all parameter groups to GPU for each underlying optimizer

    for _opt in _iter_opts(optimizers):
        if hasattr(_opt, "shard_fp32_from_float16_groups"):
            load_group_to_gpu(_opt.shard_fp32_from_float16_groups)


@torch.no_grad()
def offload_megatron_optimizer(optimizers):
    def _iter_opts(opt):
        if isinstance(opt, ChainedOptimizer):
            return opt.chained_optimizers
        return [opt]

    for _opt in _iter_opts(optimizers):
        offload_megatron_copy_params(_opt)
        opt_state_dict_values = _opt.optimizer.state.values()
        for v in opt_state_dict_values:
            if "exp_avg" in v:
                v["exp_avg"] = v["exp_avg"].to("cpu", non_blocking=True)
            if "exp_avg_sq" in v:
                v["exp_avg_sq"] = v["exp_avg_sq"].to("cpu", non_blocking=True)
        gc.collect()
        get_torch_device().empty_cache()


@torch.no_grad()
def load_megatron_optimizer(optimizers):
    def _iter_opts(opt):
        if isinstance(opt, ChainedOptimizer):
            return opt.chained_optimizers
        return [opt]

    for _opt in _iter_opts(optimizers):
        load_megatron_copy_params(_opt)
        opt_state_dict_values = _opt.optimizer.state.values()
        for v in opt_state_dict_values:
            if "exp_avg" in v:
                v["exp_avg"] = v["exp_avg"].to(get_device_id(), non_blocking=True)
            if "exp_avg_sq" in v:
                v["exp_avg_sq"] = v["exp_avg_sq"].to(get_device_id(), non_blocking=True)
        gc.collect()
        get_torch_device().empty_cache()


def get_dist_checkpoint_path(checkpoint_path):
    local_mkdir_safe(checkpoint_path)
    local_mkdir_safe(os.path.join(checkpoint_path, "dist_ckpt"))
    return os.path.join(checkpoint_path, "dist_ckpt")


def get_hf_model_checkpoint_path(checkpoint_path):
    local_mkdir_safe(checkpoint_path)
    local_mkdir_safe(os.path.join(checkpoint_path, "huggingface"))
    return os.path.join(checkpoint_path, "huggingface")


def get_transformer_config_checkpoint_path(checkpoint_path):
    os.makedirs(checkpoint_path, exist_ok=True)
    return os.path.join(checkpoint_path, "transformer_config.json")


def convert_megatron_model_to_transformers_model(
    name,
    param,
    config: PretrainedConfig,
    tp_size: int,
    num_query_groups: int,
    convert_qkv_gate_up_by_trunk_concat=False,
):
    """Convert megatron model to transformers model."""
    new_params = {}

    def convert_qkv_shard(full_tensor, q_name, k_name, v_name):
        nonlocal config
        nonlocal tp_size
        nonlocal num_query_groups

        q_shard_list = []
        k_shard_list = []
        v_shard_list = []
        hidden_size_per_head = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)

        if config.num_key_value_heads >= tp_size:
            q_size_tp = hidden_size_per_head * config.num_attention_heads // tp_size
            kv_size_tp = hidden_size_per_head * config.num_key_value_heads // tp_size
            total_size = q_size_tp + 2 * kv_size_tp
            for i in range(tp_size):
                num_query_groups_per_partition = num_query_groups // tp_size
                qkv_part = full_tensor[i * total_size : (i + 1) * total_size]
                q_size_chunk = q_size_tp // num_query_groups_per_partition
                kv_size_chunk = kv_size_tp // num_query_groups_per_partition
                for qkv_part_chunk in qkv_part.chunk(num_query_groups_per_partition):
                    q_part = qkv_part_chunk[:q_size_chunk]
                    k_part = qkv_part_chunk[q_size_chunk : q_size_chunk + kv_size_chunk]
                    v_part = qkv_part_chunk[q_size_chunk + kv_size_chunk :]
                    q_shard_list.append(q_part)
                    k_shard_list.append(k_part)
                    v_shard_list.append(v_part)
        else:
            q_size_tp = hidden_size_per_head * config.num_attention_heads // tp_size
            kv_size_tp = hidden_size_per_head
            total_size = q_size_tp + 2 * kv_size_tp
            for i in range(tp_size):
                num_query_groups_per_partition = num_query_groups // tp_size
                qkv_part = full_tensor[i * total_size : (i + 1) * total_size]
                q_size_chunk = q_size_tp // num_query_groups_per_partition
                kv_size_chunk = kv_size_tp // num_query_groups_per_partition
                for qkv_part_chunk in qkv_part.chunk(num_query_groups_per_partition):
                    q_part = qkv_part_chunk[:q_size_chunk]
                    k_part = qkv_part_chunk[q_size_chunk : q_size_chunk + kv_size_chunk]
                    v_part = qkv_part_chunk[q_size_chunk + kv_size_chunk :]
                    q_shard_list.append(q_part)
                    if i * config.num_key_value_heads % tp_size == 0:
                        k_shard_list.append(k_part)
                        v_shard_list.append(v_part)

        new_params[q_name] = torch.cat(q_shard_list, dim=0)
        new_params[k_name] = torch.cat(k_shard_list, dim=0)
        new_params[v_name] = torch.cat(v_shard_list, dim=0)

    def convert_gate_up_shard(full_tensor, gate_name, up_name):
        nonlocal config
        nonlocal tp_size

        intermediate_size_tp = config.intermediate_size // tp_size
        gate_weight_list = []
        up_weight_list = []
        for i in range(tp_size):
            gate_up_weight_tp = full_tensor[intermediate_size_tp * 2 * i : intermediate_size_tp * 2 * (i + 1)]
            gate_weight_tp = gate_up_weight_tp[:intermediate_size_tp]
            up_weight_tp = gate_up_weight_tp[intermediate_size_tp:]
            gate_weight_list.append(gate_weight_tp)
            up_weight_list.append(up_weight_tp)

        new_params[gate_name] = torch.cat(gate_weight_list, dim=0)
        new_params[up_name] = torch.cat(up_weight_list, dim=0)

    if name == "embedding.word_embeddings.weight":
        new_params["model.embed_tokens.weight"] = param
    elif "self_attention" in name:
        splitted_name = name.split(".")
        layer_number = splitted_name[2]
        component = splitted_name[4]
        param_type = splitted_name[5]
        if component == "linear_proj":
            new_params[f"model.layers.{layer_number}.self_attn.o_proj.weight"] = param
        elif component == "linear_qkv" and not isinstance(param, list):
            if param_type == "layer_norm_weight":
                new_params[f"model.layers.{layer_number}.input_layernorm.weight"] = param
            else:
                if convert_qkv_gate_up_by_trunk_concat:
                    convert_qkv_shard(
                        param,
                        f"model.layers.{layer_number}.self_attn.q_proj.{param_type}",
                        f"model.layers.{layer_number}.self_attn.k_proj.{param_type}",
                        f"model.layers.{layer_number}.self_attn.v_proj.{param_type}",
                    )
                else:
                    new_params[f"model.layers.{layer_number}.self_attn.qkv_proj.{param_type}"] = param
        elif component == "q_layernorm" or component == "k_layernorm":
            hf_component = component.replace("layer", "")
            new_params[f"model.layers.{layer_number}.self_attn.{hf_component}.weight"] = param
        else:
            assert isinstance(param, list) and len(param) == 3
            assert param_type == "weight" or param_type == "bias"
            new_params[f"model.layers.{layer_number}.self_attn.q_proj.{param_type}"] = param[0]
            new_params[f"model.layers.{layer_number}.self_attn.k_proj.{param_type}"] = param[1]
            new_params[f"model.layers.{layer_number}.self_attn.v_proj.{param_type}"] = param[2]
    elif "mlp" in name:
        splitted_name = name.split(".")
        layer_number = splitted_name[2]
        component = splitted_name[4]
        param_type = splitted_name[5]
        if component == "linear_fc1" and not isinstance(param, list):
            if param_type == "layer_norm_weight":
                new_params[f"model.layers.{layer_number}.post_attention_layernorm.weight"] = param
            elif param_type == "weight":
                if convert_qkv_gate_up_by_trunk_concat:
                    convert_gate_up_shard(
                        param,
                        f"model.layers.{layer_number}.mlp.gate_proj.weight",
                        f"model.layers.{layer_number}.mlp.up_proj.weight",
                    )
                else:
                    new_params[f"model.layers.{layer_number}.mlp.gate_up_proj.weight"] = param
        elif component == "linear_fc1" and isinstance(param, list):
            assert len(param) == 2
            assert param_type == "weight" or param_type == "bias"
            new_params[f"model.layers.{layer_number}.mlp.gate_proj.weight"] = param[0]
            new_params[f"model.layers.{layer_number}.mlp.up_proj.weight"] = param[1]
        elif component == "linear_fc2":
            new_params[f"model.layers.{layer_number}.mlp.down_proj.weight"] = param
    elif name == "decoder.final_layernorm.weight":
        new_params["model.norm.weight"] = param
    elif name == "output_layer.weight":
        new_params["lm_head.weight"] = param
    else:
        raise ValueError(f"Unknown param name: {name}")
    return new_params.keys(), new_params.values()


def broadcast_from_megatron_pp(tensor: torch.Tensor):
    # tensor is not None only in one of the pp ranks
    if tensor is not None:
        shape = tensor.shape
        dtype = tensor.dtype
        tensor_parallel = getattr(tensor, "tensor_model_parallel", None)
        partition_dim = getattr(tensor, "partition_dim", None)
        tensor_spec = (shape, dtype, tensor_parallel, partition_dim)
    else:
        tensor_spec = None
    tensor_spec_output = [None] * mpu.get_pipeline_model_parallel_world_size()
    torch.distributed.all_gather_object(
        object_list=tensor_spec_output, obj=tensor_spec, group=mpu.get_pipeline_model_parallel_group()
    )
    # find the src rank
    target_tensor_spec = None
    src_rank = None
    for rank, tensor_spec in enumerate(tensor_spec_output):
        if tensor_spec is not None:
            if target_tensor_spec is None:
                target_tensor_spec = tensor_spec
            else:
                raise ValueError("A tensor exists on two pp ranks")
            src_rank = rank
    assert target_tensor_spec is not None
    if tensor is None:
        tensor = torch.empty(size=target_tensor_spec[0], dtype=target_tensor_spec[1], device=get_device_id())
        if target_tensor_spec[2] is not None:
            tensor.tensor_model_parallel = target_tensor_spec[2]
        if target_tensor_spec[3] is not None:
            tensor.partition_dim = target_tensor_spec[3]

    global_rank = torch.distributed.get_global_rank(group=mpu.get_pipeline_model_parallel_group(), group_rank=src_rank)
    torch.distributed.broadcast(tensor=tensor, src=global_rank, group=mpu.get_pipeline_model_parallel_group())
    return tensor


def broadcast_str_from_megatron_pp(obj: Any):
    obj_output = [None] * mpu.get_pipeline_model_parallel_world_size()
    torch.distributed.all_gather_object(object_list=obj_output, obj=obj, group=mpu.get_pipeline_model_parallel_group())

    src_rank = None
    target_obj = None
    for rank, item in enumerate(obj_output):
        if item is not None:
            if target_obj is not None:
                raise ValueError("An object exists on two pp ranks")
            target_obj = item
            src_rank = rank

    assert target_obj is not None, "No valid object found to broadcast."

    global_rank = torch.distributed.get_global_rank(group=mpu.get_pipeline_model_parallel_group(), group_rank=src_rank)

    obj_output = [None] * torch.distributed.get_world_size(group=mpu.get_pipeline_model_parallel_group())
    obj_output[0] = target_obj
    torch.distributed.broadcast_object_list(
        object_list=obj_output, src=global_rank, group=mpu.get_pipeline_model_parallel_group()
    )

    return obj_output[0]


def default_tp_concat_fn(
    layer_name_mapping,
    name,
    train_params,
    infer_params,
    model_config,
    hf_config=None,
    convert_qkv_gate_up_by_simple_split=False,
):
    """
    name: name of the parameter
    train_params: training parameters
    infer_params (Iterable[torch.Tensor]): a iterator towards list of parameters all-gathered from micro_dp_group
    model_config: huggingface model_config
    TODO(zhangchi.usc1992): currently, the implementation is adhoc. We can move this function to the model
    definition so that it is model-agnostic. If the model doesn't implement this function,
    we can throw an error to force user disable TP HybridEngine.
    """
    from megatron.core import mpu

    train_tp_size = mpu.get_tensor_model_parallel_world_size()
    if layer_name_mapping.get("qkv_layer_name") in name and "layer_norm" not in name:
        # if the tensor is qkv, for each param on tp, split into q, k, v
        # concat q, k, v separately.
        q_lst = []
        k_lst = []
        v_lst = []
        num_attention_heads = model_config.num_attention_heads
        num_key_value_heads = model_config.num_key_value_heads
        if "vision_model" in name:
            num_attention_heads = hf_config.vision_config.num_heads
            num_key_value_heads = hf_config.vision_config.num_heads
        assert num_attention_heads % num_key_value_heads == 0
        num_q_per_kv = num_attention_heads // num_key_value_heads
        assert infer_params[0].shape[0] % (num_q_per_kv + 2) == 0, (
            f"param '{name}' shape '{infer_params[0].shape}' dim0 is not divisible by {num_q_per_kv + 2}"
        )
        kv_size_per_tp = infer_params[0].shape[0] // (num_q_per_kv + 2)
        split_size = [kv_size_per_tp * num_q_per_kv, kv_size_per_tp, kv_size_per_tp]
        for infer_param in infer_params:
            num_query_groups_per_partition = num_key_value_heads // train_tp_size
            for chunk in infer_param.chunk(num_query_groups_per_partition):
                split_size = [
                    kv_size_per_tp * num_q_per_kv // num_query_groups_per_partition,
                    kv_size_per_tp // num_query_groups_per_partition,
                    kv_size_per_tp // num_query_groups_per_partition,
                ]
                q, k, v = chunk.split(split_size)
                q_lst.append(q)
                k_lst.append(k)
                v_lst.append(v)
        q = torch.cat(q_lst, dim=0)
        k = torch.cat(k_lst, dim=0)
        v = torch.cat(v_lst, dim=0)
        infer_params = torch.cat((q, k, v), dim=0) if not convert_qkv_gate_up_by_simple_split else [q, k, v]

    elif (
        layer_name_mapping.get("gate_proj_layer_name") in name
        and "layer_norm" not in name
        and "vision_model.projection" not in name
    ):
        # if the tensor is gate and proj
        gate_lst = []
        up_lst = []
        for infer_param in infer_params:
            gate, up = infer_param.chunk(2)
            gate_lst.append(gate)
            up_lst.append(up)
        gate = torch.cat(gate_lst, dim=0)
        up = torch.cat(up_lst, dim=0)
        infer_params = torch.cat((gate, up), dim=0) if not convert_qkv_gate_up_by_simple_split else [gate, up]

    elif "mlp.experts.linear_fc2.weight" in name:  # moe
        infer_params = torch.cat(infer_params, dim=1)

    else:
        # concat tensor
        infer_params = torch.cat(infer_params, dim=tp_utils.get_tensor_parallel_partition_dim(train_params))

    return infer_params


def per_tensor_generator(
    actor_module,
    model_config,
    weight_converter,
    transformer_config,
    layer_name_mapping,
    convert_qkv_gate_up_by_simple_split=True,
):
    from megatron.core import parallel_state as mpu

    pp_rank = mpu.get_pipeline_model_parallel_rank()
    ep_size = mpu.get_expert_model_parallel_world_size()
    etp_size = mpu.get_expert_tensor_parallel_world_size()
    ep_group = mpu.get_expert_model_parallel_group()
    etp_group = mpu.get_expert_tensor_parallel_group()
    vpp_size = len(actor_module)
    all_gather_group = mpu.get_tensor_model_parallel_group()
    all_gather_group_size = torch.distributed.get_world_size(group=all_gather_group)

    def tensor_generator():
        for scan_vpp_idx in range(vpp_size):
            existing_keys = set()
            model = unwrap_model(actor_module[scan_vpp_idx])
            for name, param in model.named_parameters():
                existing_keys.add(name)
                yield name, param
            # note
            # there is a bug in megatron GPTModel
            # decoder.layers[n].mlp.router.expert_bias" in GPTModel is not registered in named_parameter, but in
            # state_dict(). for now we patch it by adding those keys to extra_keys.
            extra_keys = [x for x in model.state_dict().keys() if "_extra_state" not in x and x not in existing_keys]
            for name in extra_keys:
                yield name, model.state_dict()[name].to(get_device_id())

    # we need first make all rank get full model information
    meta_info = []
    for scan_vpp_idx in range(vpp_size):
        existing_keys = set()
        model = unwrap_model(actor_module[scan_vpp_idx])
        for idx, (name, _) in enumerate(model.named_parameters()):
            existing_keys.add(name)
            meta_info.append((pp_rank, scan_vpp_idx, idx, name))
        extra_keys = [x for x in model.state_dict().keys() if "_extra_state" not in x and x not in existing_keys]
        for name in extra_keys:
            meta_info.append((pp_rank, scan_vpp_idx, idx, name))

    obj_spec_output = [None] * mpu.get_pipeline_model_parallel_world_size()
    torch.distributed.all_gather_object(
        object_list=obj_spec_output, obj=meta_info, group=mpu.get_pipeline_model_parallel_group()
    )
    layer_list_meta = [item for sublist in obj_spec_output for item in sublist]

    gen_func = tensor_generator()

    # lazy load tensor for full model
    for cur_pp_rank, scan_vpp_idx, idx, name in layer_list_meta:
        if model_config.tie_word_embeddings and ("output_layers" in name):
            import warnings

            warnings.warn(
                "Current model sharing word and embedding weights, skip output layer conversion", stacklevel=2
            )
            continue

        if cur_pp_rank == pp_rank:
            try:
                cur_name, cur_tensor = next(gen_func)
            except StopIteration:
                cur_name, cur_tensor = None, None
            cur_name = normalize_model_name(name, cur_pp_rank, scan_vpp_idx, transformer_config)
        else:
            cur_tensor, cur_name = None, None

        # pp broadcast model tensor and name
        cur_name = broadcast_str_from_megatron_pp(cur_name)
        broad_pp_tensor = broadcast_from_megatron_pp(cur_tensor)

        # (xya): this is a hack to fix the name of the parameters
        while cur_name.startswith("module."):
            cur_name = cur_name[len("module.") :]

        # EP
        if ".mlp.experts.linear_fc" in cur_name and ep_size > 1:
            num_experts = weight_converter.mcore_config.num_moe_experts
            num_experts_per_rank = num_experts // ep_size
            infer_params = [torch.empty_like(broad_pp_tensor) for _ in range(ep_size)]
            torch.distributed.all_gather(infer_params, broad_pp_tensor, group=ep_group)

            name_prefix, local_expert_id = cur_name.split(".weight")
            local_expert_id = int(local_expert_id)
            global_expert_ids = [num_experts_per_rank * ep_rank + local_expert_id for ep_rank in range(ep_size)]
            global_expert_names = [f"{name_prefix}.weight{expert_id}" for expert_id in global_expert_ids]

            for name, param in zip(global_expert_names, infer_params, strict=True):
                if etp_size > 1:
                    # gather etp
                    etp_params = [torch.empty_like(param) for _ in range(etp_size)]
                    torch.distributed.all_gather(etp_params, param, group=etp_group)
                    params = etp_params
                else:
                    params = [param]

                merge_params = default_tp_concat_fn(
                    layer_name_mapping,
                    name,
                    broad_pp_tensor,
                    params,
                    model_config,
                    weight_converter.hf_config,
                    convert_qkv_gate_up_by_simple_split,
                )
                if not isinstance(merge_params, list):
                    merge_params = [merge_params]
                converted_names, converted_params = weight_converter.convert_param(name, merge_params)

                yield from zip(converted_names, converted_params, strict=True)
            continue

        # tp all gather
        if tp_utils.is_tensor_parallel_param(broad_pp_tensor):
            # allocate a new tensor with proper size
            if all_gather_group_size <= 1:
                infer_params = [broad_pp_tensor]
            else:
                infer_params = [torch.empty_like(broad_pp_tensor) for _ in range(all_gather_group_size)]
                torch.distributed.all_gather(infer_params, broad_pp_tensor, group=mpu.get_tensor_model_parallel_group())
            infer_params = default_tp_concat_fn(
                layer_name_mapping,
                cur_name,
                broad_pp_tensor,
                infer_params,
                model_config,
                weight_converter.hf_config,
                convert_qkv_gate_up_by_simple_split,
            )
        else:
            infer_params = broad_pp_tensor

        if not isinstance(infer_params, list):
            infer_params = [infer_params]
        converted_names, converted_params = weight_converter.convert_param(cur_name, infer_params)

        yield from zip(converted_names, converted_params, strict=True)


def get_transformer_layer_offset(pipeline_rank, vp_rank, config: TransformerConfig):
    '''
    Get the index offset of any pipeline stage, given the level of pipelining.

    Make pp_rank and vpp_rank as two arguments to make it more flexible,
    which is able to fetch layer offset for any pipeline stage.
    The original function only returns the layer offset for current pipeline stage.

    Extension to https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/transformer_layer.py::get_transformer_layer_offset"""
    '''
    if config.pipeline_model_parallel_size > 1:
        if (
            config.num_layers_in_first_pipeline_stage is not None
            or config.num_layers_in_last_pipeline_stage is not None
        ):
            # Calculate number of pipeline stages to distribute the remaining Transformer
            # layers after deducting the Transformer layers in the first or the last stages
            middle_pipeline_stages = config.pipeline_model_parallel_size
            middle_pipeline_stages -= sum(
                [
                    1 if x is not None else 0
                    for x in (
                        config.num_layers_in_first_pipeline_stage,
                        config.num_layers_in_last_pipeline_stage,
                    )
                ]
            )

            # Calculate layers to distribute in each pipeline stage. If the
            # num_layers_in_first_pipeline_stage and num_layers_in_last_pipeline_stage
            # are not set, we will not enable uneven pipeline. All layers will be treated
            # as middle layers.
            num_layers_in_first_pipeline_stage = (
                0 if config.num_layers_in_first_pipeline_stage is None else config.num_layers_in_first_pipeline_stage
            )
            num_layers_in_last_pipeline_stage = (
                0 if config.num_layers_in_last_pipeline_stage is None else config.num_layers_in_last_pipeline_stage
            )

            middle_num_layers = (
                config.num_layers - num_layers_in_first_pipeline_stage - num_layers_in_last_pipeline_stage
            )

            if mpu.get_virtual_pipeline_model_parallel_world_size() is not None:
                vp_size = mpu.get_virtual_pipeline_model_parallel_world_size()

                # Calculate number of layers in each virtual model chunk
                # If the num_layers_in_first_pipeline_stage and
                # num_layers_in_last_pipeline_stage are not set, all pipeline stages
                # will be treated as middle pipeline stages in the calculation
                num_layers_per_virtual_model_chunk_in_first_pipeline_stage = (
                    0
                    if config.num_layers_in_first_pipeline_stage is None
                    else config.num_layers_in_first_pipeline_stage // vp_size
                )

                num_layers_per_virtual_model_chunk_in_last_pipeline_stage = (
                    0
                    if config.num_layers_in_last_pipeline_stage is None
                    else config.num_layers_in_last_pipeline_stage // vp_size
                )

                num_layers_per_vritual_model_chunk_in_middle_pipeline_stage = middle_num_layers // vp_size

                # First stage + middle stage + last stage
                total_virtual_chunks = (
                    num_layers_per_virtual_model_chunk_in_first_pipeline_stage
                    + num_layers_per_vritual_model_chunk_in_middle_pipeline_stage
                    + num_layers_per_virtual_model_chunk_in_last_pipeline_stage
                )

                # Calculate the layer offset with interleaved uneven pipeline parallelism
                if pipeline_rank == 0:
                    offset = vp_rank * total_virtual_chunks
                else:
                    offset = (
                        vp_rank * total_virtual_chunks
                        + num_layers_per_virtual_model_chunk_in_first_pipeline_stage
                        + (pipeline_rank - 1)
                        * (num_layers_per_vritual_model_chunk_in_middle_pipeline_stage // middle_pipeline_stages)
                    )
            else:
                if middle_pipeline_stages > 0:
                    num_layers_per_pipeline_rank = middle_num_layers // middle_pipeline_stages
                else:
                    num_layers_per_pipeline_rank = 0

                middle_pipeline_rank = (
                    pipeline_rank if config.num_layers_in_first_pipeline_stage is None else pipeline_rank - 1
                )

                if pipeline_rank == 0:
                    offset = 0
                else:
                    offset = (middle_pipeline_rank * num_layers_per_pipeline_rank) + num_layers_in_first_pipeline_stage
        else:
            num_layers = config.num_layers

            # Increase the number of layers by one if we include the embedding (loss)
            # layer into pipeline parallelism partition and placement
            if config.account_for_embedding_in_pipeline_split:
                num_layers += 1

            if config.account_for_loss_in_pipeline_split:
                num_layers += 1

            num_layers_per_pipeline_rank = num_layers // config.pipeline_model_parallel_size

            if mpu.get_virtual_pipeline_model_parallel_world_size() is not None:
                vp_size = mpu.get_virtual_pipeline_model_parallel_world_size()

                num_layers_per_virtual_rank = num_layers_per_pipeline_rank // vp_size
                total_virtual_chunks = num_layers // vp_size
                offset = vp_rank * total_virtual_chunks + (pipeline_rank * num_layers_per_virtual_rank)

                # Reduce the offset of embedding layer from the total layer number
                if config.account_for_embedding_in_pipeline_split and not mpu.is_pipeline_first_stage():
                    offset -= 1
            else:
                offset = pipeline_rank * num_layers_per_pipeline_rank

                # Reduce the offset of embedding layer from the total layer number
                if config.account_for_embedding_in_pipeline_split and not mpu.is_pipeline_first_stage():
                    offset -= 1
    else:
        offset = 0
    return offset
