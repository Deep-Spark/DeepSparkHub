from time import time
import torch
from torch import nn

from colossalai.shardformer.layer import FusedMLP


class Qwen2MLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


def run_mlp(x, out_grad, func, prefix=None):
    # warm up
    for _ in range(3):
        _ = func(x)

    start = time()
    out = func(x)
    torch.cuda.synchronize()
    fwd_time = (time() - start)

    start = time()
    torch.autograd.backward(out, out_grad)
    torch.cuda.synchronize()
    bwd_time = (time() - start)

    print(f"{prefix} forward time={fwd_time:.5f}s, backward time={bwd_time:.5f}s")



if __name__ == "__main__":

    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)

    bsz = 4096
    hidden_size = 2048
    intermediate_size = 11008

    device = 'cuda:0'
    dtype = torch.bfloat16

    mlp = Qwen2MLP(hidden_size, intermediate_size).to(dtype).cuda()
    fused_mlp = FusedMLP(hidden_size, intermediate_size).to(dtype).cuda()

    x = torch.randn((bsz, hidden_size),
                    device=device,
                    dtype=dtype,
                    requires_grad=True)
    out_grad = torch.randn((bsz, hidden_size), device=device, dtype=dtype)

    print(f"dtype={dtype}")
    run_mlp(x, out_grad, mlp, prefix="mlp")
    run_mlp(x, out_grad, fused_mlp, prefix="fused_mlp")