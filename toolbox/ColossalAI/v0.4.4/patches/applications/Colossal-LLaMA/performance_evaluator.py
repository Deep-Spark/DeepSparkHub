from time import time
from typing import Optional

import torch
import torch.distributed as dist
from torch import Tensor

from colossalai.accelerator import get_accelerator
from colossalai.cluster import DistCoordinator
from colossal_llama.utils import utils

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
    tensor = torch.tensor([x], device=get_accelerator().get_current_device())
    dist.all_reduce(tensor)
    tensor = tensor / world_size
    return tensor.item()


class Timer:
    def __init__(self) -> None:
        self.start_time: Optional[float] = None
        self.duration: float = 0.0

    def start(self) -> None:
        self.start_time = time()

    def end(self) -> None:
        assert self.start_time is not None
        self.duration = time() - self.start_time
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
        model_numel: int,
        num_layers: int,
        hidden_size: int,
        vocab_size: int,
        enable_grad_checkpoint: bool = False,
        ignore_steps: int = 0,
        dp_world_size: Optional[int] = None,
    ) -> None:
        self.model_numel = model_numel
        self.enable_grad_checkpoint = enable_grad_checkpoint
        self.ignore_steps = ignore_steps
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        self.coordinator = DistCoordinator()
        self.dp_world_size = dp_world_size or self.coordinator.world_size
        self.disable: bool = False
        self.timer = Timer()
        self.num_samples: int = 0
        self.flop_megatron = 0
        self.flop: int = 0
        self.tokens_per_second_per_devices = []
        self.avg_tflops_per_gpus = []

    def on_step_start(self, step: int) -> None:
        self.disable = self.ignore_steps > 0 and step < self.ignore_steps
        self.step = step
        # if self.disable:
        #     return
        get_accelerator().synchronize()
        self.timer.start()

    def on_step_end(self, loss, inputs_size, plugin, **kwargs) -> None:
        # if self.disable:
        #     return
        get_accelerator().synchronize()
        self.timer.end()

        batch_size, seq_len = inputs_size

        self.num_samples = batch_size
        checkpoint_activations_factor = 3 + int(self.enable_grad_checkpoint)
        self.flop_megatron = (
            24 * checkpoint_activations_factor * batch_size * seq_len * self.num_layers * (self.hidden_size**2)
        ) * (
            1.0 + (seq_len / (6.0 * self.hidden_size)) + (self.vocab_size / (16.0 * self.num_layers * self.hidden_size))
        )
        self.flop = batch_size * seq_len * self.model_numel * 2 * (3 + int(self.enable_grad_checkpoint))

    # def on_fit_end(self) -> None:
        avg_duration = all_reduce_mean(self.timer.duration, self.coordinator.world_size)
        avg_throughput = self.num_samples * self.dp_world_size / (avg_duration + 1e-12)
        tokens_per_second_per_device = avg_throughput * seq_len * 2 / self.coordinator.world_size  ## BI-V150 one device has two gpus
        mp_world_size = self.coordinator.world_size // self.dp_world_size
        avg_tflops_per_gpu_megatron = self.flop_megatron / 1e12 / (avg_duration + 1e-12) / mp_world_size
        avg_tflops_per_gpu = self.flop / 1e12 / (avg_duration + 1e-12) / mp_world_size
        
        global_loss = None
        if plugin.stage_manager.is_last_stage():
            global_loss = utils.all_reduce_mean(loss, plugin)


        self.coordinator.print_on_last_process(
            f"num_samples: {self.num_samples}, dp_world_size: {self.dp_world_size}, flop_megatron: {self.flop_megatron}, flop: {self.flop}, avg_duration: {avg_duration}, "
        )
        self.coordinator.print_on_last_process(
            f"loss:{global_loss}, Throughput: {avg_throughput:.2f} samples/sec , tokens_per_second_per_device: {tokens_per_second_per_device} , TFLOPS per GPU by Megatron: {avg_tflops_per_gpu_megatron:.2f} , TFLOPS per GPU: {avg_tflops_per_gpu:.2f}"
        )

        if self.step >= self.ignore_steps and self.step < self.ignore_steps + 5:
            if self.step == self.ignore_steps + 4:
                self.coordinator.print_on_last_process("\n ---------------------------------------------" +
                                                 f"\n average values of [{self.ignore_steps} - {self.ignore_steps + 5}) steps, tokens_per_second_per_device: {sum(self.tokens_per_second_per_devices)/len(self.tokens_per_second_per_devices):.2f} , TFLOPS per GPU: {sum(self.avg_tflops_per_gpus)/len(self.avg_tflops_per_gpus):.2f} " +
                                                 "\n ---------------------------------------------")
            else:
                self.tokens_per_second_per_devices.append(tokens_per_second_per_device)
                self.avg_tflops_per_gpus.append(avg_tflops_per_gpu)
