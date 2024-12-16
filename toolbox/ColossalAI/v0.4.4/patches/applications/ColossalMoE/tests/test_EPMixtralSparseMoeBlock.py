"""
torchrun --standalone --nproc_per_node 8 test_EPMixtralSparseMoeBlock.py
"""

from copy import deepcopy

import torch
import torch.distributed as dist
from torch.testing import assert_close

from transformers.models.mixtral import MixtralConfig
from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock

import colossalai
from colossalai.booster.plugin.moe_hybrid_parallel_plugin import MoeHybridParallelPlugin
from colossalai.shardformer.modeling.mixtral import (
    EPMixtralSparseMoeBlock,
    EPOptimizeMixtralSparseMoeBlock,
)


def build_model(hidden_size, n_experts, top_k):

    torch.manual_seed(0)

    config = MixtralConfig(
        hidden_size=hidden_size,
        num_local_experts=n_experts,
        num_experts_per_tok=top_k,
        num_hidden_layers=1,
        max_position_embeddings=32768,
    )

    model = MixtralSparseMoeBlock(config).cuda()

    plugin = MoeHybridParallelPlugin(
        precision="bf16",
        tp_size=1,
        pp_size=1,
        zero_stage=1,
        ep_size=dist.get_world_size(),
    )

    ep_model = deepcopy(model)
    ep_model = EPMixtralSparseMoeBlock.from_native_module(
        ep_model,
        ep_group=plugin.ep_group,
        tp_group=plugin.tp_group,
        moe_dp_group=plugin.moe_dp_group,
    )

    return model, ep_model


if __name__ == "__main__":

    hidden_size = 16
    n_experts, top_k = 8, 2
    seq_len = 3

    rtol, atol = 1e-2, 1e-4

    colossalai.launch_from_torch(seed=0)
    model, ep_model = build_model(hidden_size, n_experts, top_k)

    # Test Forward
    x = torch.rand(1, seq_len, hidden_size, requires_grad=True).cuda()
    orig_output, orig_logits = model(x)
    ep_output, ep_logits = ep_model(x)

    assert_close(orig_logits, ep_logits, rtol=rtol, atol=atol)
    assert_close(orig_output, ep_output, rtol=rtol, atol=atol)

    # Test Backward
    orig_loss = orig_output.mean()
    orig_loss.backward()
    ep_loss = ep_output.mean()
    ep_loss.backward()
    assert_close(orig_loss, ep_loss, rtol=rtol, atol=atol)
    name_to_p = {n: p for n, p in model.named_parameters()}
    for n, ep_p in ep_model.named_parameters():
        p = name_to_p[n]
        if ep_p.grad is not None:
            assert_close(p.grad, ep_p.grad, rtol=rtol, atol=atol)

    dist.destroy_process_group()
