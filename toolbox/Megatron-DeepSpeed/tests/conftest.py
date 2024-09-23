# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import pytest

from megatron_ds.core import parallel_state
from megatron_ds.core.tensor_parallel.random import model_parallel_cuda_manual_seed

from megatron_ds.core.transformer.transformer_config import TransformerConfig

# initialize model parallel for tests
parallel_state.set_tensor_model_parallel_world_size(1)
parallel_state.set_tensor_model_parallel_rank(0)
parallel_state._set_global_memory_buffer()
parallel_state.set_pipeline_model_parallel_rank(0)
parallel_state.set_pipeline_model_parallel_world_size(1)

# model_parallel_cuda_manual_seed(123)


@pytest.fixture
def transformer_config():
    return TransformerConfig(num_layers=2, hidden_size=12, num_attention_heads=4, use_cpu_initialization=True)
