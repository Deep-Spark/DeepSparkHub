"""Megatron initialization."""
import logging
import random
import os
import time

import numpy as np
import torch
from datetime import timedelta

from megatron.legacy import fused_kernels
from megatron.training import get_adlr_autoresume
from megatron.training import get_args
from megatron.training import get_tensorboard_writer
from megatron.core import mpu, tensor_parallel
from megatron.core.rerun_state_machine import initialize_rerun_state_machine, RerunErrorInjector, RerunDiagnostic, RerunMode
from megatron.training.arguments import parse_args, validate_args
from megatron.training.yaml_arguments import validate_yaml
from megatron.training.checkpointing import load_args_from_checkpoint
from megatron.training.global_vars import set_global_variables
from megatron.core.fusions.fused_bias_dropout import bias_dropout_add_fused_train
from megatron.core.fusions.fused_bias_gelu import bias_gelu
from megatron.core.fusions.fused_bias_swiglu import bias_swiglu
from megatron.core.parallel_state import create_group
from megatron.core.utils import get_te_version, is_te_min_version, is_torch_min_version
from megatron.core import ixte_extensions
from megatron.training.initialize import (
    setup_logging,
    _set_random_seed,
    _init_autoresume,
    _initialize_tp_communicators,
    _initialize_ixte
)
from megatronspeed.core.pipeline_parallel.deepspeed_zbh1_engine import _exec_backward_only_pass, _exec_weight_pass
from megatronspeed.core.pipeline_parallel.deepspeed_zbh1_schedule import BackwardOnlyPass, WeightPass, ZeroBubbleH1Pipeline

from deepspeed.accelerator import get_accelerator
import deepspeed
from deepspeed.ops.op_builder.builder import OpBuilder

logger = logging.getLogger(__name__)


def initialize_megatron(
    extra_args_provider=None,
    args_defaults={},
    ignore_unknown_args=False,
    allow_no_cuda=False,
    skip_mpu_initialization=False,
    external_args={},
    get_embedding_ranks=None,
    get_position_embedding_ranks=None
):
    """Set global variables, initialize distributed, and
    set autoresume and random seeds.
    `allow_no_cuda` should not be set unless using megatron for cpu only
    data processing. In general this arg should not be set unless you know
    what you are doing.
    Returns a function to finalize distributed env initialization
    (optionally, only when args.lazy_mpu_init == True)
    """
    if not allow_no_cuda:
        # Make sure cuda is available.
        assert torch.cuda.is_available(), "Megatron requires CUDA."

    # Parse arguments
    args = parse_args(extra_args_provider, ignore_unknown_args)

    for key in external_args:
        if key in args:
            setattr(args, key, external_args[key])

    # Prep for checkpoint conversion.
    if args.ckpt_convert_format is not None:
        assert args.ckpt_convert_save is not None
        assert args.load is not None
        args.exit_on_missing_checkpoint = True

    if args.use_checkpoint_args or args_defaults.get("use_checkpoint_args", False):
        assert args.load is not None, "--use-checkpoints-args requires --load argument"
        assert (
            args.non_persistent_ckpt_type != "local"
        ), (
            "--use-checkpoint-args is not supported with --non_persistent_ckpt_type=local. "
            "Two-stage checkpoint loading is not implemented, and all arguments must be defined "
            "before initializing LocalCheckpointManager."
        )
        load_args_from_checkpoint(args)

    if args.yaml_cfg is not None:
        args = validate_yaml(args, args_defaults)
    else:
        validate_args(args, args_defaults)


    # set global args, build tokenizer, and set adlr-autoresume,
    # tensorboard-writer, and timers.
    set_global_variables(args)

    # set logging level
    setup_logging()

    # init rerun state
    def state_save_func():
        return {
            'rng_tracker_states': tensor_parallel.get_cuda_rng_tracker().get_states()
        }

    def state_restore_func(state_dict):
        if state_dict['rng_tracker_states']:
            tensor_parallel.get_cuda_rng_tracker().set_states(state_dict['rng_tracker_states'])

    args = get_args()
    initialize_rerun_state_machine(
        state_save_func=state_save_func,
        state_restore_func=state_restore_func,
        mode=RerunMode(args.rerun_mode),
        error_injector=RerunErrorInjector(
            error_injection_rate=args.error_injection_rate,
            error_injection_type=RerunDiagnostic(args.error_injection_type),
        ),
    )

    # torch.distributed initialization
    def finish_mpu_init():
        args = get_args()
        # Pytorch distributed.
        _initialize_distributed(get_embedding_ranks, get_position_embedding_ranks)

        # Random seeds for reproducibility.
        if args.rank == 0:
            print("> setting random seeds to {} ...".format(args.seed))
        _set_random_seed(args.seed, args.data_parallel_random_init, args.te_rng_tracker, args.inference_rng_tracker)

    if skip_mpu_initialization:
        return None

    args = get_args()
    if args.lazy_mpu_init:
        # TODO is this still a necessary option?
        args.use_cpu_initialization = True
        # delayed initialization of DDP-related stuff
        # We only set basic DDP globals
        mpu.set_tensor_model_parallel_world_size(args.tensor_model_parallel_size)
        # and return function for external DDP manager
        # to call when it has DDP initialized
        mpu.set_tensor_model_parallel_rank(args.rank)
        return finish_mpu_init
    else:
        # Megatron's MPU is the master. Complete initialization right away.
        finish_mpu_init()

        # Initialize memory buffers.
        _initialize_mem_buffs()

        # Autoresume.
        _init_autoresume()

        # Compile dependencies.
        _compile_dependencies()

        if args.tp_comm_overlap and not ixte_extensions._USE_IXTE:
            #TODO: Should this be activated with just decoder-tp-comm-overlap too?
            _initialize_tp_communicators()

        if not args.deepspeed or (args.deepspeed and args.no_pipeline_parallel):
            if ixte_extensions._USE_IXTE:
                _initialize_ixte()
        # No continuation function
        return None


def _compile_dependencies():

    args = get_args()

    # =========================
    # Compile dataset C++ code.
    # =========================
    # TODO: move this to ninja
    # LOCAL_RANK only setted when use torchrun
    if not torch.distributed.is_initialized() or int(os.environ["LOCAL_RANK"]) == 0:
        if args.deepspeed:
            start_time = time.time()
            print('> compiling dataset index builder ...')
            from megatronspeed.legacy.data.dataset_utils import compile_helper
            compile_helper()
            print('>>> done with dataset index builder. Compilation time: {:.3f} '
                'seconds'.format(time.time() - start_time), flush=True)
        else:
            start_time = time.time()
            print("> compiling dataset index builder ...")
            from megatron.core.datasets.utils import compile_helpers

            compile_helpers()
            print(
                ">>> done with dataset index builder. Compilation time: {:.3f} "
                "seconds".format(time.time() - start_time),
                flush=True,
            )

    if args.use_dataset_only:
        return
    # ==================
    # Load fused kernels
    # ==================

    # Custom kernel constraints check.
    seq_len = args.seq_length
    attn_batch_size = (
        args.num_attention_heads / args.tensor_model_parallel_size
    ) * args.micro_batch_size
    # Constraints on sequence length and attn_batch_size to enable warp based
    # optimization and upper triangular optimization (for causal mask)
    custom_kernel_constraint = (
        seq_len > 16
        and seq_len <= 16384
        and seq_len % 4 == 0
        and attn_batch_size % 4 == 0
    )
    # Print a warning.
    if not (
        (args.fp16 or args.bf16)
        and custom_kernel_constraint
        and args.masked_softmax_fusion
    ):
        if args.rank == 0:
            print(
                "WARNING: constraints for invoking optimized"
                " fused softmax kernel are not met. We default"
                " back to unfused kernel invocations.",
                flush=True,
            )

    # Always build on rank zero first.
    if torch.distributed.get_rank() == 0:
        start_time = time.time()
        print("> compiling and loading fused kernels ...", flush=True)
        fused_kernels.load(args)
        torch.distributed.barrier()
    else:
        torch.distributed.barrier()
        fused_kernels.load(args)
    # Simple barrier to make sure all ranks have passed the
    # compilation phase successfully before moving on to the
    # rest of the program. We think this might ensure that
    # the lock is released.
    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        print(
            ">>> done with compiling and loading fused kernels. "
            "Compilation time: {:.3f} seconds".format(time.time() - start_time),
            flush=True,
        )

def setup_deepspeed_random_and_activation_checkpointing(args):
    '''Optional DeepSpeed Activation Checkpointing features.
    Gives access to partition activations, contiguous memory optimizations
    and cpu checkpointing.
    Activation checkpoint requires keep track of the random states
    and setting the random seed for each MP process. Megatron uses
    mpu.get_cuda_rng_tracker and mpu.model_parallel_cuda_manual_seed
    for keeping track of the random states and setting the random seeds.
    Since they are used in places outside of activation checkpointing,
    we overwrite them to maintain consistency.
    This must be called before all the calls to mpu.model_parallel_cuda_manual_seed
    '''
    num_layers = args.num_layers // args.checkpoint_num_layers
    num_layers = num_layers if args.num_layers % args.checkpoint_num_layers == 0 else num_layers + 1
    if args.split_transformers:
        num_layers *= 2

    deepspeed.checkpointing.configure(
        mpu,
        partition_activations=args.partition_activations,
        contiguous_checkpointing=args.contigious_checkpointing,
        num_checkpoints=num_layers,
        checkpoint_in_cpu=args.checkpoint_in_cpu,
        synchronize=args.synchronize_each_layer,
        profile=args.profile_backward)

def _initialize_distributed(get_embedding_ranks, get_position_embedding_ranks):
    """Initialize torch.distributed and core model parallel."""
    args = get_args()

    device_count = torch.cuda.device_count()
    if torch.distributed.is_initialized():

        if args.rank == 0:
            print(
                "torch distributed is already initialized, "
                "skipping initialization ...",
                flush=True,
            )
        args.rank = torch.distributed.get_rank()
        args.world_size = torch.distributed.get_world_size()

    else:

        if args.rank == 0:
            print("> initializing torch distributed ...", flush=True)
        # Manually set the device ids.
        if device_count > 0:
            device = args.rank % device_count
            if args.local_rank is not None:
                assert (
                    args.local_rank == device
                ), "expected local-rank to be the same as rank % device-count."
            else:
                args.local_rank = device
            torch.cuda.set_device(device)

    if args.enable_zbh1_pipeline:
        deepspeed.runtime.pipe.schedule.TrainSchedule = ZeroBubbleH1Pipeline
        deepspeed.runtime.pipe.engine.PipelineEngine._INSTRUCTION_MAP.update(
            {
                BackwardOnlyPass: _exec_backward_only_pass,
                WeightPass: _exec_weight_pass,
            }
        )
        # Call the init process
    if args.deepspeed or args.ds_inference:
        deepspeed.init_distributed()
    else:
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend=args.distributed_backend,
                world_size=args.world_size,
                rank=args.rank,
                timeout=timedelta(minutes=args.distributed_timeout_minutes),
            )

    # Set the tensor model-parallel, pipeline model-parallel, and
    # data-parallel communicators.
    if device_count > 0:
        if mpu.model_parallel_is_initialized():
            print("model parallel is already initialized")
        else:
            if args.ds_sequence_parallel_size > 1 and args.sequence_parallel:
                raise RuntimeError(
                    f"sequence_parallel_size > 1 enables DeepSpeed's sequence parallel, "
                    f"which is not compatible with Megatron-LM's sequence parallel. "
                    f"Remove --sequence_parallel to use DeepSpeed's sequence parallel."
                )

            mpu.initialize_model_parallel(
                args.tensor_model_parallel_size,
                args.pipeline_model_parallel_size,
                args.virtual_pipeline_model_parallel_size,
                args.pipeline_model_parallel_split_rank,
                args.ds_sequence_parallel_size,
                context_parallel_size=args.context_parallel_size,
                hierarchical_context_parallel_sizes=args.hierarchical_context_parallel_sizes,
                expert_model_parallel_size=args.expert_model_parallel_size,
                num_distributed_optimizer_instances=args.num_distributed_optimizer_instances,
                expert_tensor_parallel_size=args.expert_tensor_parallel_size,
                distributed_timeout_minutes=args.distributed_timeout_minutes,
                nccl_communicator_config_path=args.nccl_communicator_config_path,
                order='tp-cp-ep-dp-pp' if not args.use_tp_pp_dp_mapping else 'tp-cp-ep-pp-dp',
                encoder_tensor_model_parallel_size=args.encoder_tensor_model_parallel_size,
                encoder_pipeline_model_parallel_size=args.encoder_pipeline_model_parallel_size,
                get_embedding_ranks=get_embedding_ranks,
                get_position_embedding_ranks=get_position_embedding_ranks,
            )
            if args.rank == 0:
                print(
                    f"> initialized tensor model parallel with size "
                    f"{mpu.get_tensor_model_parallel_world_size()}"
                )
                print(
                    f"> initialized pipeline model parallel with size "
                    f"{mpu.get_pipeline_model_parallel_world_size()}"
                )

    print(f"RANK: {torch.distributed.get_rank()}, {args.local_rank}")

    if args.deepspeed and args.deepspeed_activation_checkpointing:
        setup_deepspeed_random_and_activation_checkpointing(args)

def _initialize_mem_buffs():
    """Initialize manually allocated static memory."""
    args = get_args()
    # Initialize memory for checkpointed activations.
    if args.distribute_checkpointed_activations:
        tensor_parallel.init_checkpointed_activations_memory_buffer()

def _warmup_jit_function():
    """Compilie JIT functions before the main training steps"""
    args = get_args()
    if args.bf16:
        dtype = torch.bfloat16
    elif args.fp16:
        dtype = torch.float16
    else:
        dtype = torch.float32

    # Warmup fused bias+gelu
    bias = torch.rand(
        args.ffn_hidden_size // args.tensor_model_parallel_size,
        dtype=dtype,
        device="cuda",
    )
    input = torch.rand(
        (
            (args.seq_length // args.ds_sequence_parallel_size) if args.deepspeed else \
                (args.seq_length // args.context_parallel_size),
            args.micro_batch_size,
            args.ffn_hidden_size // args.tensor_model_parallel_size,
        ),
        dtype=dtype,
        device="cuda",
    )
    # Warmup JIT fusions with the input grad_enable state of both forward
    # prop and recomputation
    for bias_grad, input_grad in zip([True, True], [False, True]):
        bias.requires_grad, input.requires_grad = bias_grad, input_grad
        for _ in range(5):
            output = bias_gelu(bias, input)
    del bias, input, output

    # Warmup fused bias+dropout+add
    if args.sequence_parallel:
        seq_length = args.seq_length // mpu.get_tensor_model_parallel_world_size()
    else:
        seq_length = args.seq_length
    input = torch.rand(
        ((args.seq_length // args.ds_sequence_parallel_size) if args.deepspeed else \
            (args.seq_length // args.context_parallel_size), args.micro_batch_size, args.hidden_size),
        dtype=dtype,
        device="cuda",
    )
    residual = torch.rand(
        ((args.seq_length // args.ds_sequence_parallel_size) if args.deepspeed else \
            (args.seq_length // args.context_parallel_size), args.micro_batch_size, args.hidden_size),
        dtype=dtype,
        device="cuda",
    )
    bias = torch.rand((args.hidden_size), dtype=dtype, device="cuda").expand_as(
        residual
    )
    dropout_rate = 0.1
    # Warmup JIT fusions with the input grad_enable state of both forward
    # prop and recomputation
    for input_grad, bias_grad, residual_grad in zip(
        [False, True], [True, True], [True, True]
    ):
        input.requires_grad = input_grad
        bias.requires_grad = bias_grad
        residual.requires_grad = residual_grad
        for _ in range(5):
            output = bias_dropout_add_fused_train([input, bias], residual, dropout_rate)
    del bias, input, residual, output
    torch.cuda.empty_cache()
