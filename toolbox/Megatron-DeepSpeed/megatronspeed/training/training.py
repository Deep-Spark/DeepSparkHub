"""Pretrain utilities."""

import dataclasses
from datetime import datetime
import gc
import logging
import math
import os
import sys
import json
try:
    import wandb
except (ImportError, ModuleNotFoundError):
    wandb = None
from megatron.training.log_handler import CustomHandler
# Make default logging level INFO, but filter out all log messages not from MCore.
logging.basicConfig(handlers=[CustomHandler()], level=logging.INFO)
from megatron.training.theoretical_memory_usage import report_theoretical_memory
import time
# The earliest we can measure the start time.
_TRAIN_START_TIME = time.time()
import torch
from collections import OrderedDict
from enum import Enum

from megatron.core import mpu, tensor_parallel
from megatron.core.utils import (
    check_param_hashes_across_dp_replicas,
    get_model_config,
    StragglerDetector,
    is_float8tensor,
)
from megatron.training.checkpointing import load_checkpoint
from megatron.training.checkpointing import save_checkpoint
from megatron.legacy.model import Float16Module
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.distributed import DistributedDataParallel as DDP
try:
    from megatron.core.distributed import TorchFullyShardedDataParallel as torch_FSDP

    HAVE_FSDP2 = True
except ImportError:
    HAVE_FSDP2 = False

from megatron.core.distributed import finalize_model_grads
from megatron.core.enums import ModelType
from megatron.core.optimizer import get_megatron_optimizer, OptimizerConfig
from megatron.core.rerun_state_machine import (
    get_rerun_state_machine,
    destroy_rerun_state_machine,
    RerunDataIterator,
    RerunMode,
)
from megatron.training.initialize import initialize_megatron
from megatron.training.initialize import write_args_to_tensorboard
from megatron.training.initialize import set_jit_fusion_options
from megatron.legacy.data.data_samplers import build_pretraining_data_loader
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler
from megatron.core.transformer.moe import upcycling_utils
from megatron.core.transformer.moe.moe_utils import track_moe_metrics
from megatron.core.parallel_state import (
    destroy_global_memory_buffer,
    destroy_model_parallel,
)
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.core.num_microbatches_calculator import (
    destroy_num_microbatches_calculator,
    get_current_global_batch_size,
    get_current_running_global_batch_size,
    get_num_microbatches,
    update_num_microbatches)

from megatron.training.async_utils import maybe_finalize_async_save
from megatron.training.utils import (
    append_to_progress_log,
    calc_params_l2_norm,
    check_adlr_autoresume_termination,
    logical_and_across_model_parallel_group,
    reduce_max_stat_across_model_parallel_group,
    is_last_rank,
    print_rank_0,
    print_rank_last,
    report_memory,
    unwrap_model,
    update_use_dist_ckpt,
)
from megatron.training.global_vars import (
    destroy_global_vars,
    get_args,
    get_signal_handler,
    get_timers,
    get_tensorboard_writer,
    get_wandb_writer,
    get_one_logger)
from megatron.training import one_logger_utils
from megatron.training import ft_integration
from megatron.training.training import (
    preprocess_common_state_dict,
    print_datetime,
    num_floating_point_operations,
    update_train_iters,
    build_train_valid_test_data_iterators,
    get_optimizer_param_scheduler,
    build_train_valid_test_datasets,
    save_checkpoint_and_time,
    enable_forward_pre_hook,
    disable_forward_pre_hook,
    post_training_step_callbacks,
    checkpoint_and_decide_exit,
)

from megatronspeed.training.utils import is_rank_0, throughput_calculator, checkpoint_throughput_calculator, update_rotary_pos_emb
from megatronspeed.core import parallel_state

import deepspeed
from deepspeed.accelerator import get_accelerator
from deepspeed.compression.compress import init_compression, redundancy_clean
from deepspeed.runtime.data_pipeline.data_routing.helper import convert_to_random_ltd

from deepspeed import comm as dist

'''
Since v0.9.0, deepspeed.initialize() has forbidden simultaneous setting of args.deepspeed_config (Path) and ds_config dict.
So, we use ds_config dict which is the more flexible option. 
'''
def _create_ds_config_dict():
    args = get_args()
    if isinstance(args.deepspeed_config, dict) :
        ds_config_dict = args.deepspeed_config
    else:
        with open(args.deepspeed_config, 'r', encoding='utf-8') as config_file:
            ds_config_dict = json.load(config_file)

    if args.universal_checkpoint:
        ds_config_dict["checkpoint"] = {"load_universal": True}

    # Clear config path
    args.deepspeed_config = None 

    return ds_config_dict

def pretrain(train_valid_test_dataset_provider,
             model_provider,
             model_type,
             forward_step_func,
             process_non_loss_data_func=None,
             extra_args_provider=None,
             args_defaults={},
             data_post_process=None,
             external_args={},
             get_embedding_ranks=None,
             get_position_embedding_ranks=None,
             non_loss_data_func=None,):
    """Main training program.

    This function will run the followings in the order provided:
        1) initialize Megatron.
        2) setup model, optimizer and lr schedule using the model_provider.
        3) call train_val_test_data_provider to get train/val/test datasets.
        4) train the modle using the forward_step_func.

    Args:
        train_valid_test_dataset_provider: a function that takes the size of
            train/valid/test dataset and returns `train, valid, test` datasets.
        model_provider: a function that returns a vanilla version of the
            model. By vanilla we mean a simple model on cpu with no fp16 or ddp.
        model_type: an enum that specifies the type of model being trained.
        forward_step_func: a function that takes a `data iterator` and `model`,
            and returns a `loss` scalar with a dictionary with key:values being
            the info we would like to monitor during training, for example
            `lm-loss: value`. We also require that this function add
            `batch generator` to the timers class.
        process_non_loss_data_func: a function to post process outputs of the
            network. It can be used for dumping output tensors (e.g images) to
            tensorboard. It takes `collected data`(list of tensors),
            `current iteration index` and `tensorboard writer` as arguments.
        extra_args_provider: a function that takes a parser and adds arguments
            to it. It is used for programs to add their own arguments.
        args_defaults: a dictionary from argument-name to argument-value. It
            to set already parse arguments.
        get_embedding_ranks (TODO):
        get_position_embedding_ranks (TODO):
        non_loss_data_func (callable): A custom function to call during evaluation.
            It can run e.g. benchmarks.
    """

    # Initalize and get arguments, timers, and Tensorboard writer.
    initialize_megatron(extra_args_provider=extra_args_provider,
                        args_defaults=args_defaults,
                        external_args=external_args,
                        get_embedding_ranks=get_embedding_ranks,
                        get_position_embedding_ranks=get_position_embedding_ranks)

    args = get_args()
    timers = get_timers()

    if args.log_progress:
        append_to_progress_log("Starting job")

    # Initialize fault tolerance
    # NOTE: ft_integration functions other than `setup` are no-op if the FT is not initialized
    if args.enable_ft_package:
        ft_integration.setup(args)
        ft_integration.maybe_setup_simulated_fault()

    # Set pytorch JIT layer fusion options and warmup JIT functions.
    set_jit_fusion_options()

    # Adjust the startup time so it reflects the largest value.
    # This will be closer to what scheduler will see (outside of
    # image ... launches.
    global _TRAIN_START_TIME
    start_time_tensor = torch.tensor([_TRAIN_START_TIME],
                                     dtype=torch.float,
                                     device='cuda')
    torch.distributed.all_reduce(start_time_tensor,
                                 op=torch.distributed.ReduceOp.MIN)
    _TRAIN_START_TIME = start_time_tensor.item()

    app_metrics = {}
    app_metrics['app_start_time'] = round(_TRAIN_START_TIME * 1000.0)
    app_metrics['app_model_init_start_time'] = round(_TRAIN_START_TIME * 1000.0)

    print_rank_0('time to initialize megatron (seconds): {:.3f}'.format(
        time.time() - _TRAIN_START_TIME))
    print_datetime('after megatron is initialized')
    app_metrics['app_model_init_finish_time'] = one_logger_utils.get_timestamp_in_ms()

    args = get_args()
    timers = get_timers()

    if args.deepspeed:
        args.deepspeed_config_dict = _create_ds_config_dict()
        if "curriculum_learning" in args.deepspeed_config_dict and \
            "enabled" in args.deepspeed_config_dict["curriculum_learning"]:
            args.curriculum_learning_legacy = args.deepspeed_config_dict[ \
                "curriculum_learning"]["enabled"]
        if args.curriculum_learning_legacy and not args.no_pipeline_parallel:
            from deepspeed.runtime.data_pipeline.curriculum_scheduler \
                import CurriculumScheduler
            args.curriculum_scheduler = CurriculumScheduler( \
                args.deepspeed_config_dict["curriculum_learning"])
        if "compression_training" in args.deepspeed_config_dict:
            args.compression_training = True

    # Track E2E metrics on pretrain start
    one_logger_utils.on_pretrain_start()

    # Context used for persisting some state between checkpoint saves.
    if args.non_persistent_ckpt_type == 'local':
        try:
            from nvidia_resiliency_ext.checkpointing.local.ckpt_managers.local_manager import \
                LocalCheckpointManager
            from nvidia_resiliency_ext.checkpointing.local.replication.group_utils import \
                parse_group_sequence, GroupWrapper
            from nvidia_resiliency_ext.checkpointing.local.replication.strategies import \
                CliqueReplicationStrategy
        except ModuleNotFoundError:
            raise RuntimeError("The 'nvidia_resiliency_ext' module is required for local "
                               "checkpointing but was not found. Please ensure it is installed.")

        if args.replication:
            repl_strategy = CliqueReplicationStrategy.from_replication_params(
                args.replication_jump,
                args.replication_factor
            )
        else:
            repl_strategy = None

        checkpointing_context = {
            'local_checkpoint_manager': LocalCheckpointManager(args.non_persistent_local_ckpt_dir,
                                                               repl_strategy=repl_strategy
                                                               )
        }
    else:
        checkpointing_context = {}

    # Model, optimizer, and learning rate.
    timers('model-and-optimizer-setup', log_level=0).start(barrier=True)
    app_metrics['app_build_optimizer_start_time'] = one_logger_utils.get_timestamp_in_ms()
    model, optimizer, opt_param_scheduler = setup_model_and_optimizer(
        model_provider, model_type, data_post_process=data_post_process,
        build_train_valid_test_datasets_provider=train_valid_test_dataset_provider,
        checkpointing_context=checkpointing_context)

    timers('model-and-optimizer-setup').stop()
    print_datetime('after model, optimizer, and learning rate '
                   'scheduler are built')
    app_metrics['app_build_optimizer_finish_time'] = one_logger_utils.get_timestamp_in_ms()
    config = get_model_config(model[0])

    # Data stuff.
    app_metrics['app_build_dataiters_start_time'] = one_logger_utils.get_timestamp_in_ms()
    timers('train/valid/test-data-iterators-setup', log_level=0).start(
        barrier=True)
    if args.virtual_pipeline_model_parallel_size is not None:
        train_data_iterator = []
        valid_data_iterator = []
        test_data_iterator = []
        for i in range(len(model)):
            mpu.set_virtual_pipeline_model_parallel_rank(i)
            iterators = build_train_valid_test_data_iterators(
                train_valid_test_dataset_provider)
            train_data_iterator.append(iterators[0])
            valid_data_iterator.append(iterators[1])
            test_data_iterator.append(iterators[2])
    else:
        train_data_iterator, valid_data_iterator, test_data_iterator \
            = build_train_valid_test_data_iterators(
                train_valid_test_dataset_provider)
    if args.data_efficiency_curriculum_learning:
        if args.deepspeed_dataloader is not None:
            # We use args to pass the deepspeed_dataloader because adding
            # output to setup_model_and_optimizer will break the API for other
            # cases. We clear args.deepspeed_dataloader after updating
            # train_data_iterator because args will be saved in checkpoint and
            # attempting to save the whole deepspeed_dataloader will lead to
            # "AttributeError: Can't pickle local object...".
            train_data_iterator = iter(args.deepspeed_dataloader)
            args.deepspeed_dataloader = None
        else:
            train_data_iterator = None
    timers('train/valid/test-data-iterators-setup').stop()
    print_datetime('after dataloaders are built')
    app_metrics['app_build_dataiters_finish_time'] = one_logger_utils.get_timestamp_in_ms()

    # Track if training is enabled. Can only be done once args.do_train is assigned after dataloader is built.
    one_logger_utils.track_config_flags(args.train_iters, args.skip_train, args.do_train,
                                        args.do_valid, args.do_test, args.dataloader_type,
                                        args.retro_project_dir, args.retro_cyclic_train_iters)

    # Print setup timing.
    print_rank_0('done with setup ...')
    timers.log(['model-and-optimizer-setup',
                'train/valid/test-data-iterators-setup'], barrier=True)

    one_logger = get_one_logger()
    one_logger and one_logger.log_metrics(app_metrics)

    if not args.skip_train:
        print_rank_0('training ...')

        if args.dataloader_type == 'cyclic' and args.retro_project_dir:
            assert args.retro_cyclic_train_iters is not None
            args.train_iters = args.retro_cyclic_train_iters
            print_rank_0("retro cyclic train iters : %d" % args.train_iters)

        iteration = 0
        if args.do_train and args.train_iters > 0:
            iteration, num_floating_point_operations_so_far = train(
                forward_step_func,
                model, optimizer, opt_param_scheduler,
                train_data_iterator, valid_data_iterator,
                process_non_loss_data_func, config, checkpointing_context,
                non_loss_data_func)

        print_datetime('after training is done')
        # Clean the model
        if args.compression_training:
            model = [redundancy_clean(model[0], args.deepspeed_config_dict, mpu)]

        if args.save and iteration != 0 and iteration % args.save_interval != 0:
            save_checkpoint(iteration, model, optimizer, opt_param_scheduler,
                            num_floating_point_operations_so_far, checkpointing_context,
                            train_data_iterator=train_data_iterator,
                            preprocess_common_state_dict_fn=preprocess_common_state_dict)

        one_logger and one_logger.log_metrics({
            'app_train_loop_finish_time': one_logger_utils.get_timestamp_in_ms()
        })

    else:
        print_rank_0('skipping training (--skip-train is on) ...')

        iteration = args.iteration

    if args.do_valid:
        prefix = f'iteration {iteration} on validation set'
        evaluate_and_print_results(prefix, forward_step_func,
                                   valid_data_iterator, model,
                                   iteration, process_non_loss_data_func, config,
                                   verbose=True, write_to_tensorboard=not args.skip_train,
                                   non_loss_data_func=non_loss_data_func)

    if args.do_test:
        prefix = f'iteration {iteration} on test set'
        evaluate_and_print_results(prefix, forward_step_func,
                                   test_data_iterator, model,
                                   iteration, process_non_loss_data_func, config,
                                   verbose=True, write_to_tensorboard=not args.skip_train, test=True,
                                   non_loss_data_func=non_loss_data_func)

    wandb_writer = get_wandb_writer()
    if wandb_writer:
        wandb_writer.finish()

    # TODO: Is compatible with deepspeed?
    ft_integration.on_checkpointing_start()
    maybe_finalize_async_save(blocking=True)
    ft_integration.on_checkpointing_end(is_async_finalization=True)

    one_logger and one_logger.log_metrics({
        'app_finish_time': one_logger_utils.get_timestamp_in_ms()
    })
    ft_integration.shutdown()
    one_logger_utils.finish()

    print("after pretrain is done.")

    return model

def get_model(model_provider_func, model_type=ModelType.encoder_or_decoder, wrap_with_ddp=True):
    """Build the model."""
    args = get_args()
    args.model_type = model_type

    # Build model.
    if mpu.get_pipeline_model_parallel_world_size() > 1 and \
       args.virtual_pipeline_model_parallel_size is not None:
        assert model_type != ModelType.encoder_and_decoder, \
            "Interleaved schedule not supported for model with both encoder and decoder"
        model = []
        for i in range(args.virtual_pipeline_model_parallel_size):
            mpu.set_virtual_pipeline_model_parallel_rank(i)
            # Set pre_process and post_process only after virtual rank is set.
            pre_process = mpu.is_pipeline_first_stage()
            post_process = mpu.is_pipeline_last_stage()
            this_model = model_provider_func(
                pre_process=pre_process,
                post_process=post_process
            )
            this_model.model_type = model_type
            model.append(this_model)
    else:
        pre_process = mpu.is_pipeline_first_stage()
        post_process = mpu.is_pipeline_last_stage()
        add_encoder = True
        add_decoder = True
        if model_type == ModelType.encoder_and_decoder:
            if mpu.get_pipeline_model_parallel_world_size() > 1:
                rank = mpu.get_pipeline_model_parallel_rank()
                first_decoder_rank = args.encoder_pipeline_model_parallel_size
                world_size = mpu.get_pipeline_model_parallel_world_size()
                pre_process = rank == 0 or rank == first_decoder_rank
                post_process = (rank == (first_decoder_rank - 1)) or (
                        rank == (world_size - 1))
                add_encoder = mpu.is_inside_encoder(rank)
                add_decoder = mpu.is_inside_decoder(rank)
            model = model_provider_func(
                pre_process=pre_process,
                post_process=post_process,
                add_encoder=add_encoder,
                add_decoder=add_decoder)
        else:
            model = model_provider_func(
                pre_process=pre_process,
                post_process=post_process
            )
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
        print(' > number of parameters on (tensor, pipeline) '
              'model parallel rank ({}, {}): {}'.format(
            mpu.get_tensor_model_parallel_rank(),
            mpu.get_pipeline_model_parallel_rank(),
            sum([sum([p.ds_numel if hasattr(p,'ds_id') else p.nelement() for p in model_module.parameters()])
                 for model_module in model])), flush=True)

    if args.deepspeed:
        return model

    # GPU allocation.
    for model_module in model:
        model_module.cuda(torch.cuda.current_device())

    # Fp16 conversion.
    if args.fp16 or args.bf16:
        model = [Float16Module(model_module, args) for model_module in model]

    # The model_module.bfloat16()/model_module.half() above will call the inplace copy of TE's
    # Float8Tensor, which will write an unwanted value (amax calculated from the current fp8
    # param) to its amax_history. The following logic will correct the amax_history back.
    for model_module in model:
        for param in model_module.parameters():
            if is_float8tensor(param) and param._fp8_meta is not None:
                fp8_meta = param._fp8_meta['scaling_fwd']
                fp8_meta_index = param._fp8_meta_index
                if hasattr(param, 'get_high_precision_init_val'):
                    fp8_meta.amax_history[0][fp8_meta_index].copy_(
                        param.get_high_precision_init_val().abs().max()
                    )
                else:
                    fp8_meta.amax_history[0][fp8_meta_index] = 0

    if wrap_with_ddp:
        if getattr(args, "use_torch_fsdp2", False):
            assert HAVE_FSDP2, "Torch FSDP2 requires torch>=2.4.0"
            DP = torch_FSDP
        else:
            DP = DDP

        config = get_model_config(model[0])

        kwargs = {}
        for f in dataclasses.fields(DistributedDataParallelConfig):
            if hasattr(args, f.name):
                kwargs[f.name] = getattr(args, f.name)
        kwargs['grad_reduce_in_fp32'] = args.accumulate_allreduce_grads_in_fp32
        kwargs['check_for_nan_in_grad'] = args.check_for_nan_in_loss_and_grad 
        kwargs['bucket_size'] = args.ddp_bucket_size
        kwargs['average_in_collective'] = args.ddp_average_in_collective
        ddp_config = DistributedDataParallelConfig(**kwargs)

        overlap_param_gather_with_optimizer_step = getattr(args, 'overlap_param_gather_with_optimizer_step', False)
        model = [DP(config=config,
                     ddp_config=ddp_config,
                     module=model_chunk,
                     # Turn off bucketing for model_chunk 2 onwards, since communication for these
                     # model chunks is overlapped with compute anyway.
                     disable_bucketing=(model_chunk_idx > 0) or overlap_param_gather_with_optimizer_step)
                 for (model_chunk_idx, model_chunk) in enumerate(model)]

    return model

def load_model_weights_only(model_provider_func):
    """Setup model and optimizer."""
    args = get_args()
    print_rank_0('***>>>>> Args:{}'.format(args))

    model = get_model(model_provider_func)

    optimizer = None
    lr_scheduler = None

    if args.deepspeed:
        # When loading just the model weights, ZeRO can be disabled.
        if 'zero_optimization' in args.deepspeed_config_dict:
            del args.deepspeed_config_dict['zero_optimization']

        model, optimizer, _, lr_scheduler = deepspeed.initialize(
            model=model[0],
            config=args.deepspeed_config_dict
        )

        assert not isinstance(model, deepspeed.PipelineEngine), \
            'Weight loading only mode is not supported in pipeline parallelism yet.'

        model = [model]

    print_datetime('before load checkpoint')
    if args.load is not None:
        iteration, _ = load_checkpoint(model, optimizer, lr_scheduler, strict=True, load_only_weights=True)

    print_datetime('after load checkpoint weights')

    return model, optimizer, lr_scheduler

def setup_model_and_optimizer(model_provider_func,
                              model_type,
                              no_wd_decay_cond=None,
                              scale_lr_cond=None,
                              lr_mult=1.0,
                              data_post_process=None,
                              build_train_valid_test_datasets_provider=None,
                              checkpointing_context=None):
    """Setup model and optimizer."""
    args = get_args()
    timers = get_timers()
    one_logger = get_one_logger()

    model = get_model(model_provider_func, model_type)

    # initialize the compression here
    if args.compression_training:
        model, _, _, _ = deepspeed.initialize(
            model=model[0],
            args=args,
            mpu=mpu if args.no_pipeline_parallel else None,
            config=args.deepspeed_config_dict,
        )
        model = [model]
        model = [init_compression(model[0].module, args.deepspeed_config_dict, mpu)]

    unwrapped_model = unwrap_model(model)

    kwargs = {}
    for f in dataclasses.fields(OptimizerConfig):
        if hasattr(args, f.name):
            kwargs[f.name] = getattr(args, f.name)
    config = OptimizerConfig(**kwargs)
    config.timers = timers
    if args.inference:
        optimizer = None
        opt_param_scheduler = None
    else:
        optimizer = get_megatron_optimizer(config, model, no_wd_decay_cond,
                                        scale_lr_cond, lr_mult)
        # opt_param_scheduler is the old lr_scheduler plus weight decay scheduling
        opt_param_scheduler = get_optimizer_param_scheduler(optimizer)

    if args.moe_use_upcycling:
        torch.distributed.barrier()
        assert not checkpoint_exists(
            args.save
        ), ("The upcycling destination directory already exists. "
            "Please check if --moe-use-upcycling is mistakenly enabled. "
            "Upcycling should only be set for the first run when converting the dense model. "
            "All subsequent runs should remove this flag. ")
        num_experts = args.num_experts
        args.num_experts = None
        expert_model_parallel_size = args.expert_model_parallel_size
        args.expert_model_parallel_size = 1
        dense_model_for_upcycling = get_model(model_provider_func, model_type)
        args.num_experts = num_experts
        args.expert_model_parallel_size = expert_model_parallel_size
        _, args.num_floating_point_operations_so_far = upcycling_utils.load_and_upcycle_model(
            load_checkpoint,
            unwrapped_model,
            dense_model_for_upcycling,
            load_kwargs = {'model': dense_model_for_upcycling, 'optimizer': None, 'opt_param_scheduler': None}
        )
        args.iteration = 1
        save_checkpoint(args.iteration, model, None, None, args.num_floating_point_operations_so_far)
        torch.distributed.barrier()
        del dense_model_for_upcycling
        if (args.fp16 or args.bf16) and optimizer is not None:
            optimizer.reload_model_params()
        print_rank_0(f'Upcycled checkpoint saved to {args.save}')

    if args.deepspeed:
        print_rank_0("DeepSpeed is enabled.")
        pp = mpu.get_pipeline_model_parallel_world_size()
        if args.data_efficiency_curriculum_learning and build_train_valid_test_datasets_provider is not None:
            train_ds = None
            # Only need to build dataset on tp rank 0 since Megatron has the
            # broadcast_data() function that broadcast data from tp rank 0.
            if mpu.get_tensor_model_parallel_rank() == 0:
                # Number of train/valid/test samples.
                if args.train_samples:
                    train_samples = args.train_samples
                    update_train_iters(args)
                else:
                    train_samples = args.train_iters * args.global_batch_size
                # eval_iters and test_iters here are not actually used, only for
                # satisfying the input of build_train_valid_test_datasets_provider.
                # We only need to build the training data here. And we follow
                # baseline's logic to build eval/test dataset later in
                # build_train_valid_test_data_iterators.
                eval_iters = (args.train_iters // args.eval_interval + 1) * \
                            args.eval_iters
                test_iters = args.eval_iters
                train_val_test_num_samples = [train_samples,
                                            eval_iters * args.global_batch_size,
                                            test_iters * args.global_batch_size]
                # Build the datasets.
                train_ds, _, _ = build_train_valid_test_datasets_provider(
                    train_val_test_num_samples)
            model, optimizer, args.deepspeed_dataloader, opt_param_scheduler = deepspeed.initialize(
                model=model[0],
                optimizer=optimizer,
                args=args,
                lr_scheduler=opt_param_scheduler,
                training_data=train_ds,
                mpu=mpu if args.no_pipeline_parallel else None,
                config=args.deepspeed_config_dict,
            )
            model.set_data_post_process_func(data_post_process)
        else:
            model, optimizer, _, opt_param_scheduler = deepspeed.initialize(
                model=model[0],
                optimizer=optimizer,
                args=args,
                lr_scheduler=opt_param_scheduler,
                mpu=mpu if args.no_pipeline_parallel else None,
                config=args.deepspeed_config_dict,
            )
        if isinstance(model, deepspeed.PipelineEngine):
            # hack to get batch_fn from pretrain_gpt.py
            model.set_batch_fn(model.module._megatron_batch_fn)

            assert model.grid.get_pipe_parallel_rank() == mpu.get_pipeline_model_parallel_rank()
            assert model.grid.get_slice_parallel_rank() == mpu.get_tensor_model_parallel_rank()
            assert model.grid.get_data_parallel_rank() == mpu.get_data_parallel_rank()
        model = [model]

    if (args.load is not None or args.pretrained_checkpoint is not None) and not args.moe_use_upcycling:
        one_logger and one_logger.log_metrics({
            'load_checkpoint_start_time': one_logger_utils.get_timestamp_in_ms()
        })
        timers('load-checkpoint', log_level=0).start(barrier=True)
        args.iteration, args.num_floating_point_operations_so_far = load_checkpoint(
            model, optimizer, opt_param_scheduler, checkpointing_context=checkpointing_context,
                skip_load_to_model_and_opt=HAVE_FSDP2 and getattr(args, "use_torch_fsdp2", False))
        timers('load-checkpoint').stop(barrier=True)
        timers.log(['load-checkpoint'])
        one_logger and one_logger.log_metrics({
            'load_checkpoint_finish_time': one_logger_utils.get_timestamp_in_ms(),
            'load_checkpoint_time': timers('load-checkpoint').active_time()
        })
    else:
        args.iteration = 0
        args.num_floating_point_operations_so_far = 0

    # get model without FP16 and/or DDP wrappers
    if args.iteration == 0 and len(unwrapped_model) == 1 \
        and hasattr(unwrapped_model[0], 'init_state_dict_from_bert'):
        print_rank_0("Initializing ICT from pretrained BERT model")
        unwrapped_model[0].init_state_dict_from_bert()
        if args.fp16:
            optimizer.reload_model_params()

    # Convert checkpoint format.
    if args.ckpt_convert_format is not None:
        load_ckpt_format = args.ckpt_format
        args.ckpt_format = args.ckpt_convert_format
        args.save = os.path.join(args.ckpt_convert_save, args.ckpt_convert_format)
        update_use_dist_ckpt(args)

        save_checkpoint(args.iteration, model, optimizer, opt_param_scheduler,
                        args.num_floating_point_operations_so_far,
                        preprocess_common_state_dict_fn=preprocess_common_state_dict)

        print_rank_0("> converted checkpoint: %s -> %s." % (load_ckpt_format, args.ckpt_format))
        torch.distributed.barrier()
        exit()

    return model, optimizer, opt_param_scheduler

def train_step(forward_step_func, data_iterator,
               model, optimizer, opt_param_scheduler, config):
    """Single training step."""
    args = get_args()
    timers = get_timers()

    if args.deepspeed and args.ds_pipeline_enabled:
        skipped_iter = 0
        num_zeros_in_grad = 0
        assert isinstance(model[0], deepspeed.PipelineEngine)
        loss = model[0].train_batch(data_iter=data_iterator)
        additional_losses = model[0].get_additional_losses()
        loss_key = 'lm loss' if additional_losses is None else 'loss'  # use "lm loss" for backward compatibility
        loss_dict = OrderedDict({loss_key: loss})
        if additional_losses is not None:
            loss_dict.update(additional_losses)
        grad_norm = model[0].get_global_grad_norm()
        return loss_dict, skipped_iter, False, False, 0, grad_norm, num_zeros_in_grad

    # TODO: Is compatible with deepspeed?
    rerun_state_machine = get_rerun_state_machine()
    while rerun_state_machine.should_run_forward_backward(data_iterator):
        # Set grad to zero.
        if not args.deepspeed:
            for model_chunk in model:
                model_chunk.zero_grad_buffer()
            optimizer.zero_grad()

        # Forward pass.
        forward_backward_func = get_forward_backward_func()
        losses_reduced = forward_backward_func(
            forward_step_func=forward_step_func,
            data_iterator=data_iterator,
            model=model,
            num_microbatches=get_num_microbatches(),
            seq_length=args.seq_length,
            micro_batch_size=args.micro_batch_size,
            decoder_seq_length=args.decoder_seq_length,
            forward_only=False)
    should_checkpoint, should_exit, exit_code = rerun_state_machine.should_checkpoint_and_exit()
    if should_exit:
        return {}, True, should_checkpoint, should_exit, exit_code, None, None

    # Empty unused memory.
    if args.empty_unused_memory_level >= 1:
        torch.cuda.empty_cache()

    # Vision gradients.
    if getattr(args, 'vision_pretraining', False) and args.vision_pretraining_type == "dino":
        unwrapped_model = unwrap_model(model[0])
        unwrapped_model.cancel_gradients_last_layer(args.curr_iteration)

    # Update parameters.
    timers('optimizer', log_level=1).start(barrier=args.barrier_with_L1_time)
    if args.deepspeed:
        increment = get_num_microbatches() * \
                    args.micro_batch_size * \
                    args.data_parallel_size
        model[0].step(lr_kwargs={'increment': increment})
        update_successful = model[0].was_step_applied()
    else:
        update_successful, grad_norm, num_zeros_in_grad = optimizer.step()
    timers('optimizer').stop()

    if not args.deepspeed:
        # when freezing sub-models we may have a mixture of successful and unsucessful ranks,
        # so we must gather across mp ranks
        update_successful = logical_and_across_model_parallel_group(update_successful)
        # grad_norm and num_zeros_in_grad will be None on ranks without trainable params,
        # so we must gather across mp ranks
        grad_norm = reduce_max_stat_across_model_parallel_group(grad_norm)
        if args.log_num_zeros_in_grad:
            num_zeros_in_grad = reduce_max_stat_across_model_parallel_group(num_zeros_in_grad)

    # Vision momentum.
    if getattr(args, 'vision_pretraining', False) and args.vision_pretraining_type == "dino":
        unwrapped_model = unwrap_model(model[0])
        unwrapped_model.update_momentum(args.curr_iteration)

    # Update learning rate.
    if args.deepspeed:
        skipped_iter = 0
        grad_norm = None
        num_zeros_in_grad = None
        loss_reduced = {}
        for key in losses_reduced[0]:
            losses_reduced_for_key = [x[key] for x in losses_reduced]
            loss_reduced[key] = sum(losses_reduced_for_key) / len(losses_reduced_for_key)
        return loss_reduced, skipped_iter, False, False, 0, grad_norm, num_zeros_in_grad
    else:
        if update_successful:
            increment = get_num_microbatches() * \
                        args.micro_batch_size * \
                        args.data_parallel_size
            opt_param_scheduler.step(increment=increment)
            skipped_iter = 0
        else:
            skipped_iter = 1

        # Empty unused memory.
        if args.empty_unused_memory_level >= 2:
            torch.cuda.empty_cache()

        if mpu.is_pipeline_last_stage(ignore_virtual=True):
            # Average loss across microbatches.
            loss_reduced = {}
            for key in losses_reduced[0].keys():
                numerator = 0
                denominator = 0
                for x in losses_reduced:
                    val = x[key]
                    # there is one dict per microbatch. in new reporting, we average
                    # over the total number of tokens across the global batch.
                    if isinstance(val, tuple) or isinstance(val, list):
                        numerator += val[0]
                        denominator += val[1]
                    else:
                        # legacy behavior. we average over the number of microbatches,
                        # and so the denominator is 1.
                        numerator += val
                        denominator += 1
                loss_reduced[key] = numerator / denominator
            return loss_reduced, skipped_iter, should_checkpoint, should_exit, exit_code, grad_norm, num_zeros_in_grad
    return {}, skipped_iter, should_checkpoint, should_exit, exit_code, grad_norm, num_zeros_in_grad

def training_log(loss_dict, total_loss_dict, learning_rate, decoupled_learning_rate, iteration,
                 loss_scale, report_memory_flag, skipped_iter,
                 grad_norm, params_norm, num_zeros_in_grad,
                 model=None, optimizer=None):
    """Log training information such as losses, timing, ...."""
    args = get_args()
    timers = get_timers()
    writer = get_tensorboard_writer()
    wandb_writer = get_wandb_writer()
    one_logger = get_one_logger()

    # 获取 Iluvatar 设备判断
    # IS_BI_V150 = "BI-V150" in execCmd("ixsmi -L")
    IS_BI_V150 = True

    # Advanced, skipped, and Nan iterations.
    advanced_iters_key = 'advanced iterations'
    skipped_iters_key = 'skipped iterations'
    nan_iters_key = 'nan iterations'
    # Advanced iterations.
    if not skipped_iter:
        total_loss_dict[advanced_iters_key] = total_loss_dict.get(
            advanced_iters_key, 0) + 1
    else:
        if advanced_iters_key not in total_loss_dict:
            total_loss_dict[advanced_iters_key] = 0
    # Skipped iterations.
    total_loss_dict[skipped_iters_key] = total_loss_dict.get(
        skipped_iters_key, 0) + skipped_iter
    # Update losses and set nan iterations
    got_nan = False
    for key in loss_dict:
        if not skipped_iter:
            total_loss_dict[key] = total_loss_dict.get(
                key, torch.tensor([0.0], dtype=torch.float, device='cuda')) + loss_dict[key]
        else:
            value = loss_dict[key].float().sum().item()
            is_nan = value == float('inf') or \
                     value == -float('inf') or \
                     value != value
            got_nan = got_nan or is_nan
    total_loss_dict[nan_iters_key] = total_loss_dict.get(
        nan_iters_key, 0) + int(got_nan)

    # Logging.
    timers_to_log = [
        'forward-backward',
        'forward-compute',
        'backward-compute',
        'batch-generator',
        'forward-recv',
        'forward-send',
        'backward-recv',
        'backward-send',
        'forward-send-forward-recv',
        'forward-send-backward-recv',
        'backward-send-forward-recv',
        'backward-send-backward-recv',
        'forward-backward-send-forward-backward-recv',
        'layernorm-grads-all-reduce',
        'embedding-grads-all-reduce',
        'all-grads-sync',
        'params-all-gather',
        'optimizer-copy-to-main-grad',
        'optimizer-unscale-and-check-inf',
        'optimizer-clip-main-grad',
        'optimizer-count-zeros',
        'optimizer-inner-step',
        'optimizer-copy-main-to-model-params',
        'optimizer']

    # Calculate batch size.
    batch_size = args.micro_batch_size * args.data_parallel_size * \
        get_num_microbatches()

    # Track app tag & app tag ID
    one_logger_utils.track_app_tag(batch_size, args.world_size, args.seq_length)

    total_iterations = total_loss_dict[advanced_iters_key] + \
                       total_loss_dict[skipped_iters_key]

    # learning rate will be None on ranks without trainable params, so we must gather across mp ranks
    learning_rate = reduce_max_stat_across_model_parallel_group(learning_rate)
    # Tensorboard values.
    # Timer requires all the ranks to call.
    if args.log_timers_to_tensorboard and \
       (iteration % args.tensorboard_log_interval == 0):
        timers.write(timers_to_log, writer, iteration,
                     normalizer=total_iterations)
    if writer and (iteration % args.tensorboard_log_interval == 0):
        if wandb_writer:
            wandb_writer.log({'samples vs steps': args.consumed_train_samples},
                             iteration)
        writer.add_scalar('learning-rate', learning_rate, iteration)
        writer.add_scalar('learning-rate vs samples', learning_rate,
                            args.consumed_train_samples)
        if wandb_writer:
            wandb_writer.log({'learning-rate': learning_rate}, iteration)
        if args.decoupled_lr is not None:
            writer.add_scalar('decoupled-learning-rate', decoupled_learning_rate, iteration)
        if args.skipped_train_samples > 0:
            writer.add_scalar('skipped-train-samples', args.skipped_train_samples, iteration)
            if wandb_writer:
                wandb_writer.log({'skipped-train-samples': args.skipped_train_samples}, iteration)

        writer.add_scalar('batch-size', batch_size, iteration)
        writer.add_scalar('batch-size vs samples', batch_size,
                            args.consumed_train_samples)
        if wandb_writer:
            wandb_writer.log({'batch-size': batch_size}, iteration)
        for key in loss_dict:
            writer.add_scalar(key , loss_dict[key], iteration)
            writer.add_scalar(key + ' vs samples', loss_dict[key],
                              args.consumed_train_samples)
            if wandb_writer:
                wandb_writer.log({key: loss_dict[key]}, iteration)
        if args.log_loss_scale_to_tensorboard:
            writer.add_scalar('loss-scale', loss_scale, iteration)
            writer.add_scalar('loss-scale vs samples', loss_scale,
                              args.consumed_train_samples)
            if wandb_writer:
                wandb_writer.log({'loss-scale': loss_scale}, iteration)
        if args.log_world_size_to_tensorboard:
            writer.add_scalar('world-size', args.world_size, iteration)
            writer.add_scalar('world-size vs samples', args.world_size,
                              args.consumed_train_samples)
            if wandb_writer:
                wandb_writer.log({'world-size': args.world_size}, iteration)
        if grad_norm is not None:
            writer.add_scalar('grad-norm', grad_norm, iteration)
            writer.add_scalar('grad-norm vs samples', grad_norm,
                              args.consumed_train_samples)
            if wandb_writer:
                wandb_writer.log({'grad-norm': grad_norm}, iteration)
        if num_zeros_in_grad is not None:
            writer.add_scalar('num-zeros', num_zeros_in_grad, iteration)
            writer.add_scalar('num-zeros vs samples', num_zeros_in_grad,
                              args.consumed_train_samples)
            if wandb_writer:
                wandb_writer.log({'num-zeros': num_zeros_in_grad}, iteration)
        if params_norm is not None:
            writer.add_scalar('params-norm', params_norm, iteration)
            writer.add_scalar('params-norm vs samples', params_norm,
                              args.consumed_train_samples)
            if wandb_writer:
                wandb_writer.log({'params-norm': params_norm}, iteration)
        if args.log_memory_to_tensorboard:
            mem_stats = torch.cuda.memory_stats()
            writer.add_scalar(
                "mem-reserved-bytes",
                mem_stats["reserved_bytes.all.current"],
                iteration,
            )
            writer.add_scalar(
                "mem-allocated-bytes",
                mem_stats["allocated_bytes.all.current"],
                iteration,
            )
            writer.add_scalar(
                "mem-max-allocated-bytes",
                mem_stats["allocated_bytes.all.peak"],
                iteration,
            )
            writer.add_scalar(
                "mem-allocated-count",
                mem_stats["allocation.all.current"],
                iteration,
            )
    if args.num_experts is not None:
        moe_loss_scale = 1 / get_num_microbatches()
        track_moe_metrics(moe_loss_scale, iteration, writer, wandb_writer, total_loss_dict, args.moe_per_layer_logging)

    if iteration % args.log_interval == 0:
        elapsed_time = timers('interval-time').elapsed(barrier=True)
        elapsed_time_per_iteration = elapsed_time / total_iterations
        seq_len = args.seq_length
        if hasattr(args, 'actual_seq_length'):
            seq_len = args.actual_seq_length
        samples_per_sec, tflops, approx_parameters_in_billions = throughput_calculator(
            model,
            args,
            elapsed_time,
            total_iterations
        )
        samples_per_sec_per_replica = samples_per_sec / args.data_parallel_size
        tokens_per_sec = samples_per_sec * seq_len
        tokens_per_sec_per_replica = tokens_per_sec / args.data_parallel_size
        tokens_per_gpu_per_second = tokens_per_sec / args.world_size
        tokens_per_gpu_per_second_per_replica = tokens_per_gpu_per_second / args.data_parallel_size
        if wandb is not None and getattr(wandb, 'run', None) is not None:
            assert wandb.run is not None
            wandb_metrics = {
                'throughput/iteration-time': elapsed_time_per_iteration,  # 1000 ms / s
                'throughput/samples_per_sec': samples_per_sec,
                'throughput/samples_per_sec_per_replica': samples_per_sec_per_replica,
                'throughput/tokens_per_sec': tokens_per_sec,
                'throughput/tokens_per_sec_per_replica': tokens_per_sec_per_replica,
                'throughput/tokens_per_gpu_per_sec': tokens_per_gpu_per_second,
                'throughput/tokens_per_gpu_per_sec_per_replica': tokens_per_gpu_per_second_per_replica,
                'throughput/tflops': tflops,
                'throughput/approx_params_in_billions': approx_parameters_in_billions,
                'throughput/elapsed_ms_per_iteration': elapsed_time_per_iteration,
                'throughput/iteration': iteration,
            }
            if loss_dict is not None:
                wandb_metrics |= {
                    f'loss/{k}': v for k, v in loss_dict.items()
                }
                wandb_metrics |= {'loss/iteration': iteration}

        throughput = num_floating_point_operations(args, batch_size) / (
            elapsed_time_per_iteration * 10**12 * args.world_size)

        one_logger_utils.track_e2e_metrics(args.log_throughput, throughput)

        if args.log_timers_to_tensorboard:
            if writer:
                writer.add_scalar('iteration-time',
                                  elapsed_time_per_iteration, iteration)
            if wandb_writer:
                wandb_writer.log({'iteration-time': elapsed_time_per_iteration},
                                 iteration)
        log_string = f" [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]"
        log_string += ' iteration {:8d}/{:8d} |'.format(
            iteration, args.train_iters)
        log_string += ' consumed samples: {:12d} |'.format(
            args.consumed_train_samples)
        log_string += ' consumed tokens: {:12d} |'.format(
            args.consumed_train_tokens)
        if args.skipped_train_samples > 0:
            log_string += ' skipped samples: {:12d} |'.format(
                args.skipped_train_samples)
        log_string += ' elapsed time per iteration (ms): {:.1f} |'.format(
            elapsed_time_per_iteration * 1000.0)
        log_string += ' tokens per second: {:.2f} |'.format(
            batch_size * total_iterations * args.seq_length / elapsed_time)
        if IS_BI_V150:
            log_string += ' tokens per second per device: {:.2f} |'.format(
                batch_size * total_iterations * args.seq_length * 2 / args.world_size / elapsed_time) # BI-V150 one device has two gpus
        else:
            log_string += ' tokens per second per device: {:.2f} |'.format(
                batch_size * total_iterations * args.seq_length / args.world_size / elapsed_time)
        if args.log_throughput:
            log_string += f' throughput per GPU (TFLOP/s/GPU): {throughput:.1f} |'
            if args.log_timers_to_tensorboard:
                if writer:
                    writer.add_scalar('throughput', throughput, iteration)
                if wandb_writer:
                    wandb_writer.log({'throughput': throughput}, iteration)

        # Decoupled_learning_rate should be not None only on first and last pipeline stage.
        log_string += ' learning rate: {:.6E} |'.format(learning_rate)
        if args.decoupled_lr is not None and (mpu.is_pipeline_first_stage(ignore_virtual=True) or
                                              mpu.is_pipeline_last_stage(ignore_virtual=True)):
            assert decoupled_learning_rate is not None
            log_string += ' decoupled learning rate: {:.6E} |'.format(decoupled_learning_rate)
        else:
            assert decoupled_learning_rate is None
        log_string += ' global batch size: {:5d} |'.format(batch_size)
        if wandb is not None and getattr(wandb, 'run', None) is not None:
            wandb_metrics |= {
                'training/iteration': iteration,
                'training/iteration_time': elapsed_time_per_iteration,
                'training/iteration_time_vs_tokens': (
                    (elapsed_time_per_iteration
                        / args.consumed_train_tokens)
                ),
                'training/iteration_time_vs_samples': (
                    (elapsed_time_per_iteration
                        / args.consumed_train_samples),
                ),
                'training/consumed_samples': args.consumed_train_samples,
                'training/consumed_tokens': args.consumed_train_tokens,
            }
        for key in total_loss_dict:
            if key not in [advanced_iters_key, skipped_iters_key,
                           nan_iters_key]:
                avg = total_loss_dict[key].item() / \
                      float(max(1, total_loss_dict[advanced_iters_key]))
                if avg > 0.0:
                    log_string += ' {}: {:.6E} |'.format(key, avg)
                total_loss_dict[key] = torch.tensor([0.0], dtype=torch.float, device='cuda')
        if wandb is not None and getattr(wandb, 'run', None) is not None:
            wandb.log(wandb_metrics)
        if loss_scale is not None:
            log_string += ' loss scale: {:.1f} |'.format(loss_scale)
        if grad_norm is not None:
            log_string += ' grad norm: {:.3f} |'.format(grad_norm)
        if num_zeros_in_grad is not None:
            log_string += ' num zeros: {:.1f} |'.format(num_zeros_in_grad)
        if params_norm is not None:
            log_string += ' params norm: {:.3f} |'.format(params_norm)
        log_string += ' actual seqlen: {:5d} |'.format(seq_len)
        log_string += ' number of skipped iterations: {:3d} |'.format(
            total_loss_dict[skipped_iters_key])
        log_string += ' number of nan iterations: {:3d} |'.format(
            total_loss_dict[nan_iters_key])
        log_string += ' samples per second: {:.3f} |'.format(samples_per_sec)
        # log_string += ' tokens per gpu per second (tgs): {:.3f} |'.format(tokens_per_gpu_per_second)
        log_string += ' TFLOPs: {:.2f} |'.format(tflops)
        total_loss_dict[advanced_iters_key] = 0
        total_loss_dict[skipped_iters_key] = 0
        total_loss_dict[nan_iters_key] = 0
        print_rank_last(log_string)
        ## 计算一些step的平均值
        global elapsed_time_per_iteration_10
        global tokens_per_second_10
        global tflops_10
        global tps_per_device
        global times
        log_initialized = False
        if not log_initialized:
            elapsed_time_per_iteration_10 = 0.0
            tokens_per_second_10 = 0.0
            tflops_10 = 0.0
            times = 0
            tps_per_device = 0.0
            log_initialized = True
        if iteration >= 1:
            elapsed_time_per_iteration_10 += elapsed_time_per_iteration * 1000.0
            tokens_per_second_10 += batch_size * total_iterations * args.seq_length / elapsed_time
            tflops_10 += tflops
            if IS_BI_V150:
                tps_per_device += (batch_size * total_iterations * args.seq_length * 2 / args.world_size / elapsed_time)
            else:
                tps_per_device += (batch_size * total_iterations * args.seq_length / args.world_size / elapsed_time)
            times += 1
        if times == 5:
            print_rank_last(f"---------------------------------------------------------------")
            print_rank_last(f"Some iteration elapsed time per iteration (ms):{elapsed_time_per_iteration_10/times} | tokens per second: {tokens_per_second_10/times} | tokens per second per device: {tps_per_device/times} | TFLOPs: {tflops_10/times}")
            print_rank_last(f"---------------------------------------------------------------")
        if report_memory_flag:
            # Report memory after optimizer state has been initialized.
            if torch.distributed.get_rank() == 0:
                num_microbatches = get_num_microbatches()
                report_theoretical_memory(args, num_microbatches=num_microbatches, verbose=True)
            report_memory('(after {} iterations)'.format(iteration))
            report_memory_flag = False
        if not args.disable_logging:
            timers.log(timers_to_log, normalizer=args.log_interval)

    return report_memory_flag

def train(forward_step_func, model, optimizer, opt_param_scheduler,
          train_data_iterator, valid_data_iterator,
          process_non_loss_data_func, config, checkpointing_context, non_loss_data_func):
    """Training function: run train_step desired number of times, run validation, checkpoint."""
    args = get_args()
    timers = get_timers()
    one_logger = get_one_logger()

    if torch.distributed.get_rank() == 0:
        print("config: ", config)

    # Write args to tensorboard
    write_args_to_tensorboard()

    # Turn on training mode which enables dropout.
    for model_module in model:
        model_module.train()

    # Tracking loss.
    total_loss_dict = {}

    # Iterations.
    iteration = args.iteration

    # Track E2E metrics at the start of training
    one_logger_utils.on_train_start(iteration=iteration, consumed_train_samples=args.consumed_train_samples,
                                    train_samples=args.train_samples, seq_length=args.seq_length,
                                    train_iters=args.train_iters, save=args.save, async_save=args.async_save,
                                    log_throughput=args.log_throughput,
                                    num_floating_point_operations_so_far=args.num_floating_point_operations_so_far)

    num_floating_point_operations_so_far = args.num_floating_point_operations_so_far

    # Setup some training config params
    if not args.deepspeed:
        config.grad_scale_func = optimizer.scale_loss
    config.timers = timers
    if isinstance(model[0], DDP) and args.overlap_grad_reduce:
        assert config.no_sync_func is None, \
            ('When overlap_grad_reduce is True, config.no_sync_func must be None; '
             'a custom no_sync_func is not supported when overlapping grad-reduce')
        config.no_sync_func = [model_chunk.no_sync for model_chunk in model]
        if len(model) == 1:
            config.no_sync_func = config.no_sync_func[0]
        if args.align_grad_reduce:
            config.grad_sync_func = [model_chunk.start_grad_sync for model_chunk in model]
            if len(model) == 1:
                config.grad_sync_func = config.grad_sync_func[0]
    if args.overlap_param_gather and args.align_param_gather:
        config.param_sync_func = [model_chunk.start_param_sync for model_chunk in model]
        if len(model) == 1:
            config.param_sync_func = config.param_sync_func[0]
    if not args.deepspeed:
        config.finalize_model_grads_func = finalize_model_grads

    timers('interval-time', log_level=0).start(barrier=True)
    print_datetime('before the start of training step')
    report_memory_flag = True
    pre_hook_enabled = False
    should_exit = False
    exit_code = 0

    if args.manual_gc:
        # Disable the default garbage collector and perform the collection manually.
        # This is to align the timing of garbage collection across ranks.
        assert args.manual_gc_interval >= 0, \
            'Manual garbage collection interval should be larger than or equal to 0.'
        gc.disable()
        gc.collect()

    # Singleton initialization of straggler detector.
    if args.log_straggler:
        global stimer
        world = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        mmcnt = args.straggler_minmax_count
        stimer.configure(world, rank,
                mmcnt = mmcnt,
                enabled = not args.disable_straggler_on_startup,
                port = args.straggler_ctrlr_port)
    num_floating_point_operations_since_last_log_event = 0.0

    num_microbatches = get_num_microbatches()
    eval_duration = 0.0
    eval_iterations = 0

    def get_e2e_base_metrics():
        """Get base metrics values for one-logger to calculate E2E tracking metrics.
        """
        num_floating_point_operations_since_current_train_start = \
            num_floating_point_operations_so_far - args.num_floating_point_operations_so_far
        return {
            'iteration': iteration,
            'train_duration': timers('interval-time').active_time(),
            'eval_duration': eval_duration,
            'eval_iterations': eval_iterations,
            'total_flops_since_current_train_start': num_floating_point_operations_since_current_train_start,
            'num_floating_point_operations_so_far': num_floating_point_operations_so_far,
            'consumed_train_samples': args.consumed_train_samples,
            'world_size': args.world_size,
            'seq_length': args.seq_length
        }
    # Cache into one-logger for callback
    if one_logger:
        with one_logger.get_context_manager():
            one_logger.store_set('get_e2e_base_metrics', get_e2e_base_metrics)

    prof = None
    if args.profile and torch.distributed.get_rank() in args.profile_ranks and args.use_pytorch_profiler:
        prof = torch.profiler.profile(
        schedule=torch.profiler.schedule(
            wait=max(args.profile_step_start-1, 0),
            warmup=1 if args.profile_step_start > 0 else 0,
            active=args.profile_step_end-args.profile_step_start,
            repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(args.tensorboard_dir),
        record_shapes=True,
        with_stack=True)
        prof.start()

    start_iteration = iteration
    # Disable forward pre-hook to start training to ensure that errors in checkpoint loading
    # or random initialization don't propagate to all ranks in first all-gather (which is a
    # no-op if things work correctly).
    if args.use_distributed_optimizer and args.overlap_param_gather:
        disable_forward_pre_hook(model, param_sync=False)
        # Also remove param_sync_func temporarily so that sync calls made in
        # `forward_backward_func` are no-ops.
        param_sync_func = config.param_sync_func
        config.param_sync_func = None
        pre_hook_enabled = False
    # Also, check weight hash across DP replicas to be very pedantic.
    if args.check_weight_hash_across_dp_replicas_interval is not None:
        assert check_param_hashes_across_dp_replicas(model, cross_check=True), \
            "Parameter hashes not matching across DP replicas"
        torch.distributed.barrier()
        print_rank_0(f">>> Weight hashes match after {iteration} iterations...")

    # Run training iterations till done.
    while iteration < args.train_iters:
        if args.profile and torch.distributed.get_rank() in args.profile_ranks:
            if args.use_pytorch_profiler:
                prof.step()
            elif iteration == args.profile_step_start:
                torch.cuda.cudart().cudaProfilerStart()
                torch.autograd.profiler.emit_nvtx(record_shapes=True).__enter__()

        ft_integration.on_checkpointing_start()
        maybe_finalize_async_save(blocking=False)
        ft_integration.on_checkpointing_end(is_async_finalization=True)

        # Update number of microbatches first without consistency check to decide if a
        # checkpoint should be saved. If the number of microbatches is different
        # from the previous iteration, save a checkpoint. Then run consistency check
        # to make sure training configuration is still valid.
        update_num_microbatches(args.consumed_train_samples, consistency_check=False, verbose=True)
        if get_num_microbatches() != num_microbatches and iteration != 0:
            assert get_num_microbatches() > num_microbatches, \
                (f"Number of microbatches should be increasing due to batch size rampup; "
                 f"instead going from {num_microbatches} to {get_num_microbatches()}")
            if args.save is not None:
                save_checkpoint_and_time(iteration, model, optimizer,
                                        opt_param_scheduler,
                                        num_floating_point_operations_so_far,
                                        checkpointing_context, train_data_iterator=train_data_iterator)
        num_microbatches = get_num_microbatches()
        update_num_microbatches(args.consumed_train_samples, consistency_check=True, verbose=True)
        if args.deepspeed:
            # inform deepspeed of any batch size changes
            global_batch_size = mpu.get_data_parallel_world_size() * \
                                args.micro_batch_size * \
                                get_num_microbatches()
            model[0].set_train_batch_size(global_batch_size)

        if args.curriculum_learning_legacy and not args.no_pipeline_parallel:
            curriculum_seqlen = args.curriculum_scheduler.update_difficulty( \
                    args.iteration + 1)
            if iteration == 0 or curriculum_seqlen != args.curriculum_seqlen:
                if args.use_rotary_position_embeddings:
                    update_rotary_pos_emb(curriculum_seqlen)
            args.curriculum_seqlen = curriculum_seqlen

        # Run training step.
        args.curr_iteration = iteration
        ft_integration.on_training_step_start()
        loss_dict, skipped_iter, should_checkpoint, should_exit, exit_code, grad_norm, num_zeros_in_grad = \
            train_step(forward_step_func,
                       train_data_iterator,
                       model,
                       optimizer,
                       opt_param_scheduler,
                       config)
        ft_integration.on_training_step_end()
        if should_checkpoint:
            save_checkpoint_and_time(iteration, model, optimizer,
                                     opt_param_scheduler,
                                     num_floating_point_operations_so_far,
                                     checkpointing_context, train_data_iterator=train_data_iterator)
        if should_exit:
            break

        # Enable forward pre-hooks after first set of forward and backward passes.
        # When running in fp16, skip all NaN iterations until steady-state loss scaling value
        # is reached.
        if iteration == start_iteration:
            if skipped_iter:
                # Only enable forward pre-hook after a training step has successfully run. Relevant
                # for fp16 codepath where first XX iterations are skipped until steady-state loss
                # scale value is reached.
                start_iteration = iteration + 1
            else:
                # Enable forward pre-hook after training step has successfully run. All subsequent
                # forward passes will use the forward pre-hook / `param_sync_func` in
                # `forward_backward_func`.
                if args.use_distributed_optimizer and args.overlap_param_gather:
                    enable_forward_pre_hook(model)
                    config.param_sync_func = param_sync_func
                    pre_hook_enabled = True

        iteration += 1
        args.iteration = iteration
        batch_size = mpu.get_data_parallel_world_size() * \
                     args.micro_batch_size * \
                     get_num_microbatches()
        args.consumed_train_samples += batch_size
        num_skipped_samples_in_batch = (get_current_global_batch_size() -
                                        get_current_running_global_batch_size())
        if args.decrease_batch_size_if_needed:
            assert num_skipped_samples_in_batch >= 0
        else:
            assert num_skipped_samples_in_batch == 0
        args.skipped_train_samples += num_skipped_samples_in_batch
        num_floating_point_operations_in_batch = num_floating_point_operations(args, batch_size)
        num_floating_point_operations_so_far += num_floating_point_operations_in_batch
        num_floating_point_operations_since_last_log_event += num_floating_point_operations_in_batch

        # This actual_seq_length is used for actual consumed tokens calculation, flops calculation, and logging.
        args.actual_seq_length = args.seq_length
        if args.curriculum_learning_legacy or args.data_efficiency_curriculum_learning:
            args.actual_seq_length = args.curriculum_seqlen
        if args.curriculum_learning_legacy or args.data_efficiency_curriculum_learning:
            if hasattr(args, 'data_efficiency_curriculum_learning_numel'):
                act_mbsz = args.data_efficiency_curriculum_learning_numel / args.curriculum_seqlen
                act_token = act_mbsz * args.actual_seq_length
                args.consumed_train_tokens += mpu.get_data_parallel_world_size() * \
                        get_num_microbatches() * act_token
            else:
                args.consumed_train_tokens += batch_size * args.actual_seq_length
        else:
            args.consumed_train_tokens += batch_size * args.actual_seq_length

        # Logging.
        if args.deepspeed:
            if hasattr(model[0].optimizer, 'cur_scale'):
                loss_scale = model[0].optimizer.cur_scale
            else:
                loss_scale = None
        else:
            if not optimizer.is_stub_optimizer:
                loss_scale = optimizer.get_loss_scale().item()
            else:
                loss_scale = 1.0
        params_norm = None
        if args.log_params_norm:
            params_norm = calc_params_l2_norm(model)

        learning_rate = None
        decoupled_learning_rate = None
        for param_group in optimizer.param_groups:
            if param_group['is_decoupled_lr']:
                decoupled_learning_rate = param_group['lr']
            else:
                learning_rate = param_group['lr']
        report_memory_flag = training_log(loss_dict, total_loss_dict,
                                          learning_rate,
                                          decoupled_learning_rate,
                                          iteration, loss_scale,
                                          report_memory_flag, skipped_iter,
                                          grad_norm, params_norm, num_zeros_in_grad,
                                          model, optimizer)

        # Evaluation
        if args.eval_interval and iteration % args.eval_interval == 0 and \
           args.do_valid:
            timers('interval-time').stop()
            if args.use_distributed_optimizer and args.overlap_param_gather:
                disable_forward_pre_hook(model)
                pre_hook_enabled = False
            if args.manual_gc and args.manual_gc_eval:
                # Collect all objects.
                gc.collect()
            prefix = f'iteration {iteration}'
            timers('eval-time', log_level=0).start(barrier=True)
            evaluate_and_print_results(prefix, forward_step_func,
                                       valid_data_iterator, model,
                                       iteration, process_non_loss_data_func,
                                       config, verbose=False, write_to_tensorboard=True,
                                       non_loss_data_func=non_loss_data_func)
            eval_duration += timers('eval-time').elapsed()
            eval_iterations += args.eval_iters
            timers('eval-time').stop()
            one_logger_utils.track_e2e_metrics()

            if args.manual_gc and args.manual_gc_eval:
                # Collect only the objects created and used in evaluation.
                gc.collect(generation=0)
            if args.use_distributed_optimizer and args.overlap_param_gather:
                enable_forward_pre_hook(model)
                pre_hook_enabled = True
            timers('interval-time', log_level=0).start(barrier=True)

        # Miscellaneous post-training-step functions (e.g., FT heartbeats, GC).
        # Some of these only happen at specific iterations.
        post_training_step_callbacks(model, optimizer, opt_param_scheduler, iteration, prof,
                                     num_floating_point_operations_since_last_log_event)

        # Checkpoint and decide whether to exit.
        should_exit = checkpoint_and_decide_exit(model, optimizer, opt_param_scheduler, iteration,
                                                 num_floating_point_operations_so_far,
                                                 checkpointing_context, train_data_iterator)
        if should_exit:
            break

    one_logger_utils.track_e2e_metrics()

    # Flush TensorBoard, WandB writers and one-logger
    writer = get_tensorboard_writer()
    if writer:
        writer.flush()

    # Close out pre-hooks if using distributed optimizer and overlapped param gather.
    if pre_hook_enabled:
        disable_forward_pre_hook(model)

    ft_integration.on_checkpointing_start()
    maybe_finalize_async_save(blocking=True)
    ft_integration.on_checkpointing_end(is_async_finalization=True)

    # If any exit conditions (signal handler, duration, iterations) have been reached, exit.
    if should_exit:
        wandb_writer = get_wandb_writer()
        if wandb_writer:
            wandb_writer.finish()
        ft_integration.shutdown()
        sys.exit(exit_code)

    return iteration, num_floating_point_operations_so_far

def evaluate(forward_step_func,
             data_iterator,
             model,
             process_non_loss_data_func,
             config,
             verbose=False,
             non_loss_data_func=None):
    """Evaluation."""
    args = get_args()
    timers = get_timers()

    timers('evaluate', log_level=0).start(barrier=True)

    if args.vision_pretraining and args.vision_pretraining_type == "dino":
        from megatron.legacy.model.vision.knn_monitor import compute_feature_bank
        compute_feature_bank(model)

    # Turn on evaluation mode which disables dropout.
    for model_module in model:
        model_module.eval()

    # Disable result validation during evaluation
    rerun_state_machine = get_rerun_state_machine()
    rerun_mode = rerun_state_machine.get_mode()
    rerun_state_machine.set_mode(RerunMode.DISABLED)

    if args.curriculum_learning_legacy and not args.no_pipeline_parallel:
        # When curriculum learning is used with pipeline parallelism, we need
        # this logic to ensure that the eval data is not truncated. If there
        # is a seqlen change due to that, we need to call
        # reset_activation_shape() to reset some buffers in deepspeed pipeline
        # engine.
        if args.curriculum_seqlen < args.seq_length:
            args.curriculum_seqlen = args.seq_length
            if args.use_rotary_position_embeddings:
                update_rotary_pos_emb(args.curriculum_seqlen)
            model[0].reset_activation_shape()

    total_loss_dict = {}

    # make validation batch size independent from training batch size
    eval_batch_size = args.global_batch_size
    eval_num_microbatches = eval_batch_size // \
        (args.micro_batch_size * args.data_parallel_size)

    with torch.no_grad():
        iteration = 0
        if verbose:
            print_rank_0(f'Evaluating on {args.eval_iters * eval_batch_size} samples')
        while iteration < args.eval_iters:
            iteration += 1
            if verbose:
                print_rank_0(f'Evaluating iter {iteration}/{args.eval_iters}')

            forward_backward_func = get_forward_backward_func()
            # Don't care about timing during evaluation
            config.timers = None
            if args.deepspeed and args.ds_pipeline_enabled:
                # DeepSpeed uses eval_batch() and already aggregates losses.
                assert isinstance(model, list) and len(model) == 1
                loss = model[0].eval_batch(data_iterator)
                loss_dicts = [{'lm loss' : loss}] * get_num_microbatches()
            else:
                ft_integration.on_eval_step_start()
                loss_dicts = forward_backward_func(
                    forward_step_func=forward_step_func,
                    data_iterator=data_iterator,
                    model=model,
                    num_microbatches=eval_num_microbatches,
                    seq_length=args.seq_length,
                    micro_batch_size=args.micro_batch_size,
                    decoder_seq_length=args.decoder_seq_length,
                    forward_only=True)
                ft_integration.on_eval_step_end()
            config.timers = get_timers()

            # Empty unused memory
            if args.empty_unused_memory_level >= 1:
                torch.cuda.empty_cache()

            if mpu.is_pipeline_last_stage(ignore_virtual=True):
                # Reduce across processes.
                for loss_dict in loss_dicts:
                    for key in loss_dict:
                        if key not in total_loss_dict:
                            total_loss_dict[key] = torch.tensor([0.0, 0.0], dtype=torch.float).cuda()
                        val = loss_dict[key]
                        if isinstance(val, tuple) or isinstance(val, list):
                            total_loss_dict[key][0] += val[0]
                            total_loss_dict[key][1] += val[1]
                        else:
                            total_loss_dict[key][0] += val
                            total_loss_dict[key][1] += 1

            args.consumed_valid_samples += eval_batch_size

            if args.exit_duration_in_mins:
                train_time = (time.time() - _TRAIN_START_TIME) / 60.0
                done_cuda = torch.tensor(
                    [train_time > args.exit_duration_in_mins],
                    dtype=torch.int, device='cuda')
                torch.distributed.all_reduce(
                    done_cuda, op=torch.distributed.ReduceOp.MAX)
                done = done_cuda.item()
                if done:
                    rerun_state_machine.set_mode(rerun_mode)
                    print_rank_0('Exiting during evaluation, timelimit reached')
                    return None, None, True

        collected_non_loss_data = None
        if non_loss_data_func is not None:
            collected_non_loss_data = non_loss_data_func(model)
        elif process_non_loss_data_func is not None and is_last_rank():
            collected_non_loss_data = forward_backward_func(
                forward_step_func=forward_step_func,
                data_iterator=data_iterator,
                model=model,
                num_microbatches=get_num_microbatches(),
                seq_length=args.seq_length,
                micro_batch_size=args.micro_batch_size,
                decoder_seq_length=args.decoder_seq_length,
                forward_only=True,
                collect_non_loss_data=True)

    # Move model back to the train mode.
    for model_module in model:
        model_module.train()

    for key in total_loss_dict:
        numerator, denominator = total_loss_dict[key]
        total_loss_dict[key] = numerator / denominator

    timers('evaluate').stop()
    timers.log(['evaluate'])

    if args.curriculum_learning_legacy and not args.no_pipeline_parallel:
        # roll back to actual curriculum seqlen at the end of eval.
        args.curriculum_seqlen = args.curriculum_scheduler.update_difficulty( \
            args.iteration + 1)
        if args.curriculum_seqlen < args.seq_length:
            if args.use_rotary_position_embeddings:
                update_rotary_pos_emb(args.curriculum_seqlen)
            model[0].reset_activation_shape()

    rerun_state_machine.set_mode(rerun_mode)

    rerun_state_machine.set_mode(rerun_mode)

    return total_loss_dict, collected_non_loss_data, False

def evaluate_and_print_results(prefix, forward_step_func,
                               data_iterator, model,
                               iteration, process_non_loss_data_func, config,
                               verbose=False, write_to_tensorboard=True, test=False, non_loss_data_func=None):
    """Helper function to evaluate and dump results on screen."""
    args = get_args()
    if write_to_tensorboard:
        writer = get_tensorboard_writer()
    else:
        writer = None

    wandb_writer = get_wandb_writer()

    total_loss_dict, collected_non_loss_data, timelimit = evaluate(
        forward_step_func, data_iterator, model,
        process_non_loss_data_func, config, verbose, non_loss_data_func)
    # Timelimit hit during evaluation
    if timelimit:
        return
    string = f' validation loss at {prefix} | '
    for key in total_loss_dict:
        string += '{} value: {:.6E} | '.format(key, total_loss_dict[key].item())
        ppl = math.exp(min(20, total_loss_dict[key].item()))
        string += '{} PPL: {:.6E} | '.format(key, ppl)
        if writer:
            writer.add_scalar('{} validation'.format(key),
                              total_loss_dict[key].item(),
                              iteration)
            writer.add_scalar('{} validation vs samples'.format(key),
                              total_loss_dict[key].item(),
                              args.consumed_train_samples)
            if args.log_validation_ppl_to_tensorboard:
                writer.add_scalar('{} validation ppl'.format(key), ppl,
                                  iteration)
                writer.add_scalar('{} validation ppl vs samples'.format(key),
                                  ppl, args.consumed_train_samples)
            if wandb_writer and is_last_rank():
                wandb_writer.log({
                    '{} validation'.format(key): total_loss_dict[key].item()},
                    iteration)

    if process_non_loss_data_func is not None and writer and is_last_rank():
        process_non_loss_data_func(collected_non_loss_data, iteration, writer)

    length = len(string) + 1
    print_rank_last('-' * length)
    print_rank_last(string)
    print_rank_last('-' * length)

def build_train_valid_test_data_loaders(
        build_train_valid_test_datasets_provider):
    """Build pretraining data loaders."""

    args = get_args()

    (train_dataloader, valid_dataloader, test_dataloader) = (None, None, None)

    print_rank_0('> building train, validation, and test datasets ...')

    # Backward compatibility, assume fixed batch size.
    if args.iteration > 0 and args.consumed_train_samples == 0:
        assert args.train_samples is None, \
            'Only backward compatiblity support for iteration-based training'
        args.consumed_train_samples = args.iteration * args.global_batch_size
    if args.iteration > 0 and args.consumed_valid_samples == 0:
        if args.train_samples is None:
            args.consumed_valid_samples = (args.iteration // args.eval_interval) * \
                args.eval_iters * args.global_batch_size

    # Rely on distributed-aware core datasets, temporary
    is_distributed = getattr(build_train_valid_test_datasets_provider, "is_distributed", False)

    # Construct the data pipeline
    ds_sequence_parallel = parallel_state.get_sequence_parallel_world_size() > 1 or args.force_ds_sequence_parallel
    rank_in_parallel_group = parallel_state.get_sequence_parallel_rank() if ds_sequence_parallel else mpu.get_tensor_model_parallel_rank()
    if is_distributed or rank_in_parallel_group == 0:

        # Build datasets.
        train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
            build_train_valid_test_datasets_provider)
        # Build dataloders.
        train_dataloader = build_pretraining_data_loader(
            train_ds, args.consumed_train_samples)
        if args.skip_train:
            valid_dataloader = build_pretraining_data_loader(valid_ds, 0)
        else:
            valid_dataloader = build_pretraining_data_loader(
                valid_ds, args.consumed_valid_samples)
        test_dataloader = build_pretraining_data_loader(test_ds, 0)

        # Flags to know if we need to do training/validation/testing.
        do_train = train_dataloader is not None and args.train_iters > 0
        do_valid = valid_dataloader is not None and args.eval_iters > 0
        do_test = test_dataloader is not None and args.eval_iters > 0
        flags = torch.tensor(
            [int(do_train), int(do_valid), int(do_test)],
            dtype=torch.long, device='cuda')
    else:
        flags = torch.tensor([0, 0, 0], dtype=torch.long, device='cuda')

    torch.distributed.broadcast(flags, 0)

    args.do_train = getattr(args, "do_train", False) or flags[0].item()
    args.do_valid = getattr(args, "do_valid", False) or flags[1].item()
    args.do_test = getattr(args, "do_test", False) or flags[2].item()

    return train_dataloader, valid_dataloader, test_dataloader
