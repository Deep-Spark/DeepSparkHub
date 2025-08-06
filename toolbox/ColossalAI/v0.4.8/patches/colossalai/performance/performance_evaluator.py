from time import time
from typing import Optional

import torch
import torch.distributed as dist
from torch import Tensor
import torch.distributed
from torch.profiler import ProfilerActivity, profile, schedule, tensorboard_trace_handler

from colossalai.cluster import DistCoordinator
from colossalai.utils import get_current_device


def divide(x: float, y: float) -> float:
    if y == 0:
        return float("inf")
    elif y == float("inf"):
        return float("nan")
    return x / y


@torch.no_grad()
def all_reduce_mean(x: float, world_size: int) -> float:
    if world_size == 1:
        return x
    # BUG: RuntimeError: Invalid scalar type when use dist.all_reduce(tensor, group=gloo_group)
    # # Use CPU tensor to avoid OOM/weird NCCl error
    # gloo_group = dist.new_group(backend="gloo")
    # tensor = torch.tensor([x], device="cpu")
    # dist.all_reduce(tensor, group=gloo_group)
    # tensor = tensor / world_size
    # return tensor.item()

    tensor = torch.tensor([x], device=get_current_device(), dtype=torch.float)
    dist.all_reduce(tensor)
    tensor = tensor / world_size
    return tensor.item()


def get_profile_context(enable_flag, warmup_steps, active_steps, save_dir, nsys=False):
    class DummyProfiler:
        def __init__(self):
            self.step_number = 0

        def step(self):
            self.step_number += 1

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            pass

    class NsysProfiler:
        def __init__(self, warmup_steps, active_steps):
            self.step_number = 0
            self.warmup_steps = warmup_steps
            self.active_steps = active_steps

        def step(self):
            if self.step_number == self.warmup_steps:
                torch.cuda.cudart().cudaProfilerStart()
            elif self.step_number == self.warmup_steps + self.active_steps:
                torch.cuda.cudart().cudaProfilerStop()
            self.step_number += 1

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            pass

    if enable_flag:
        if nsys:
            return NsysProfiler(warmup_steps, active_steps)

        return profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(wait=0, warmup=warmup_steps, active=active_steps),
            on_trace_ready=tensorboard_trace_handler(save_dir),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )
    else:
        return DummyProfiler()


class Timer:
    def __init__(self) -> None:
        self.start_time: Optional[float] = None
        self.duration: float = 0.0

    def start(self) -> None:
        self.start_time = time()

    def end(self) -> None:
        assert self.start_time is not None
        self.duration += time() - self.start_time
        self.start_time = None

    def reset(self) -> None:
        self.duration = 0.0


class PerformanceEvaluator:
    """
        Callback for valuate the performance of the model.
    Args:
        actor_num_params: The number of parameters of the actor model.
        critic_num_params: The number of parameters of the critic model.
        initial_model_num_params: The number of parameters of the initial model.
        reward_model_num_params: The number of parameters of the reward model.
        enable_grad_checkpoint: Whether to enable gradient checkpointing.
        ignore_episodes: The number of episodes to ignore when calculating the performance.
    """

    def __init__(
        self,
        dp_world_size: int,
        num_layers: int,
        hidden_size: int,
        vocab_size: int,
        seq_len: int,
        batch_size: int,
        ignore_steps: int = 0,
    ) -> None:
        self.dp_world_size = dp_world_size

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.batch_size = batch_size

        self.ignore_steps = ignore_steps

        self.coordinator = DistCoordinator()
        self.timer = Timer()
        self.disable = False

        self.num_samples = 0
        self.global_batch_size = batch_size * self.dp_world_size

    def on_step_start(self, step: int) -> None:
        self.disable = self.ignore_steps > 0 and step < self.ignore_steps
        if self.disable:
            return

        self.timer.start()

    def on_step_end(self) -> None:
        if self.disable:
            return

        self.timer.end()
        self.num_samples += self.global_batch_size

    def on_fit_end(self) -> None:
        avg_duration = all_reduce_mean(self.timer.duration, self.coordinator.world_size)
        avg_samples_per_step = self.num_samples / (avg_duration + 1e-12)

        # BI-V150 per GPU device has two cards
        avg_tokens_per_gpu = avg_samples_per_step * self.seq_len * 2 / self.coordinator.world_size

        self.coordinator.print_on_master(
            f"global_batch_size: {self.global_batch_size}, batch_size: {self.batch_size}, dp_world_size: {self.dp_world_size}"
        )
        self.coordinator.print_on_master(
            f"Tokens per GPU per Second: {avg_tokens_per_gpu:.2f}, Average Samples Per Step: {avg_samples_per_step:.2f} samples/s"
        )
