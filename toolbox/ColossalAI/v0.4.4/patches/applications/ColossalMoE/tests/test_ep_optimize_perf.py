"""
torchrun --standalone --nproc_per_node 4 test_ep_optimize_perf.py
"""

from time import time
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


def build_model():

    torch.manual_seed(0)

    config = MixtralConfig(max_position_embeddings=32768)
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

    ep_optimize_model = deepcopy(model)
    ep_optimize_model = EPOptimizeMixtralSparseMoeBlock.from_native_module(
        ep_optimize_model,
        ep_group=plugin.ep_group,
        tp_group=plugin.tp_group,
        moe_dp_group=plugin.moe_dp_group,
    )

    return model, ep_model, ep_optimize_model


def rum_model(input, model, model_prefix=None):
    start = time()
    output, logits = model(input)
    loss = output.mean()
    torch.cuda.synchronize()
    fwd_end = time()
    fwd_time = fwd_end - start

    loss.backward()
    torch.cuda.synchronize()
    bwd_end = time()
    bwd_time = bwd_end - fwd_end
    if dist.get_rank() == 0:
        print(
            f"{model_prefix} model forward time={fwd_time:.3f}s, backward time={bwd_time:.3f}s"
        )

    return output, fwd_time, bwd_time


def warm_up(input, model):
    output, logits = model(input)
    loss = output.mean()
    loss.backward()


if __name__ == "__main__":

    n_warm_up = 10
    seq_len, hidden_size = 1024, 4096

    colossalai.launch_from_torch(seed=0)
    model, ep_model, ep_opt_model = build_model()

    input = torch.rand(1, seq_len, hidden_size, requires_grad=True).cuda()
    for _ in range(n_warm_up):
        warm_up(input, model)
        warm_up(input, ep_model)
        warm_up(input, ep_opt_model)
    torch.cuda.synchronize()

    orig_fwd_out, _, _ = rum_model(input, model, model_prefix="Original")
    ep_fwd_out, fwd_time, bwd_time = rum_model(input, ep_model, model_prefix="EP")
    ep_opt_fwd_out, opt_fwd_time, opt_bwd_time = rum_model(
        input, ep_opt_model, model_prefix="EP_Optimize"
    )

    if dist.get_rank() == 0:
        print(f"ep forward improve {100*fwd_time/opt_fwd_time:.0f}%")
        print(f"ep backward improve {100*bwd_time/opt_bwd_time:.0f}%")
        print(
            f"ep forward+backward improve {100*(fwd_time+bwd_time)/(opt_fwd_time+opt_bwd_time):.0f}%"
        )

    rtol, atol = 1e-2, 1e-4

    # Test Forward Accuracy
    assert_close(orig_fwd_out, ep_opt_fwd_out, rtol=rtol, atol=atol)

    # Test Backward Accuracy
    name_to_p = {n: p for n, p in model.named_parameters()}
    for n, ep_p in ep_opt_model.named_parameters():
        p = name_to_p[n]
        if ep_p.grad is not None:
            assert_close(p.grad, ep_p.grad, rtol=rtol, atol=atol)

    dist.destroy_process_group()
