"""Input/output checkpointing."""

from logging import getLogger
import os
import random
import sys
import numpy as np
from time import time
from functools import wraps

import torch

from megatron.core import mpu, tensor_parallel, dist_checkpointing
from megatron.core.dist_checkpointing.mapping import ShardedObject
from megatron.core.dist_checkpointing.serialization import get_default_load_sharded_strategy
from megatron.core.dist_checkpointing.strategies.fully_parallel import \
    FullyParallelSaveStrategyWrapper, FullyParallelLoadStrategyWrapper
from megatron.core.num_microbatches_calculator import update_num_microbatches
from megatron.training.async_utils import schedule_async_save
from megatron.training.global_vars import get_args, get_one_logger, get_tokenizer
from megatron.training.utils import unwrap_model, print_rank_0, append_to_progress_log, is_last_rank
from megatron.core.dist_checkpointing.serialization import \
    get_default_save_sharded_strategy
from megatron.training.one_logger_utils import on_save_checkpoint_start, on_save_checkpoint_success

from megatron.training.checkpointing import (
    set_checkpoint_version,
    get_checkpoint_version,
    get_checkpoint_name,
    get_distributed_optimizer_checkpoint_name,
    get_checkpoint_tracker_filename,
    checkpoint_exists,
    read_metadata,
    get_rng_state,
    _load_base_checkpoint,
    fix_query_key_value_ordering,
    ensure_directory_exists,
    find_checkpoint_rank_0,
    logger
)

from deepspeed.checkpoint import (
    ORIGINAL_VOCAB_SIZE,
    PADDED_VOCAB_SIZE,
    UNIVERSAL_CHECKPOINT_INFO,
    UNIVERSAL_CHECKPOINT_VERSION_KEY,
    UNIVERSAL_CHECKPOINT_VERSION_VALUE,
)

# [ModelOpt]: Import
try:
    from modelopt.torch.opt.plugins import (
        save_modelopt_state,
        save_sharded_modelopt_state,
        restore_modelopt_state,
        restore_sharded_modelopt_state,
    )
    has_nvidia_modelopt = True
except Exception:
    has_nvidia_modelopt = False

_CHECKPOINT_VERSION = None

def check_checkpoint_args(checkpoint_args):
    """Ensure fixed arguments for a model are the same for the input
    arguments and the one retrieved from checkpoint."""
    args = get_args()

    def _compare(arg_name, old_arg_name=None, default=None):
        if old_arg_name is not None:
            ckpt_arg_name = old_arg_name
        else:
            ckpt_arg_name = arg_name
        if default is not None:
            checkpoint_value = getattr(checkpoint_args, ckpt_arg_name, default)
        else:
            checkpoint_value = getattr(checkpoint_args, ckpt_arg_name)
        args_value = getattr(args, arg_name)
        error_message = '{} value from checkpoint ({}) is not equal to the ' \
                        'input argument value ({}).'.format(
                            arg_name, checkpoint_value, args_value)
        assert checkpoint_value == args_value, error_message

    _compare('num_layers')
    _compare('hidden_size')
    _compare('num_attention_heads')
    _compare('add_position_embedding', default=True)
    if args.vocab_file:
        _compare('max_position_embeddings')
        if not args.universal_checkpoint:
            _compare('make_vocab_size_divisible_by')
        if not args.use_dist_ckpt or not args.universal_checkpoint:
            _compare('padded_vocab_size')
        _compare('tokenizer_type')
    if args.data_parallel_random_init:
        _compare('data_parallel_random_init')
    if get_checkpoint_version() < 3.0 and not args.universal_checkpoint:
        _compare('tensor_model_parallel_size',
                 old_arg_name='model_parallel_size')
    if get_checkpoint_version() >= 3.0 and (not args.use_dist_ckpt or not args.universal_checkpoint):
        _compare('tensor_model_parallel_size')
        _compare('pipeline_model_parallel_size')

def save_checkpoint(iteration, model, optimizer, opt_param_scheduler, num_floating_point_operations_so_far=0, checkpointing_context=None,
                    pipeline_rank=None,expert_rank=None, tensor_rank=None, pipeline_parallel=None, expert_parallel=None):
    """Save a model checkpoint.

    Checkpointing context is used to persist some checkpointing state
    throughout a single job. Must be initialized externally (not used if None).
    """
    start_ckpt = time()
    args = get_args()

    # Prepare E2E metrics at start of save checkpoint
    productive_metrics = on_save_checkpoint_start(args.async_save)

    # Only rank zero of the data parallel writes to the disk.
    if not args.deepspeed:
        model = unwrap_model(model)

    ckpt_format = args.dist_ckpt_format if args.use_dist_ckpt else 'torch'
    print_rank_0('saving checkpoint at iteration {:7d} to {} in {} format'.format(
        iteration, args.save, ckpt_format))

    # Collect rng state across data parallel ranks.
    rng_state = get_rng_state(args.use_dist_ckpt)

    # Checkpoint name.
    checkpoint_name = get_checkpoint_name(args.save, iteration, release=False, pipeline_parallel=pipeline_parallel,
        tensor_rank=tensor_rank, pipeline_rank=pipeline_rank, expert_parallel=expert_parallel, expert_rank=expert_rank, return_base_dir=args.use_dist_ckpt)

    # Save distributed optimizer's custom parameter state.
    if args.use_distributed_optimizer and not args.no_save_optim and optimizer is not None and not args.use_dist_ckpt:
        optim_checkpoint_name = \
            get_distributed_optimizer_checkpoint_name(checkpoint_name)
        ensure_directory_exists(optim_checkpoint_name)
        optimizer.save_parameter_state(optim_checkpoint_name)

    async_save_request = None
    if args.async_save:
        if not args.use_dist_ckpt:
            raise NotImplementedError('Async checkpoint save not implemented for legacy checkpoints')
        elif args.dist_ckpt_format != 'torch_dist':
            raise NotImplementedError(f'Async checkpoint save not implemented for {args.dist_ckpt_format} distributed checkpoint format')

    # Collect args, model, RNG.
    if not torch.distributed.is_initialized() \
            or mpu.get_data_modulo_expert_parallel_rank(with_context_parallel=True) == 0 \
            or args.use_dist_ckpt or args.deepspeed:

        optim_sd_kwargs = {}
        if args.use_dist_ckpt and args.use_distributed_optimizer:
            optim_sd_kwargs['sharding_type'] = ('fully_sharded_model_space'
                                                if args.ckpt_fully_parallel_save
                                                else 'dp_zero_gather_scatter')
            print_rank_0(f'Storing distributed optimizer sharded state of type {optim_sd_kwargs["sharding_type"]}')
        state_dict = generate_state_dict(args, model, optimizer, opt_param_scheduler, rng_state,
                                         args.use_dist_ckpt, iteration, optim_sd_kwargs=optim_sd_kwargs)

        if not args.deepspeed:
            state_dict['num_floating_point_operations_so_far'] = num_floating_point_operations_so_far
            if args.use_dist_ckpt:
                if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                    ensure_directory_exists(checkpoint_name, check_parent=False)
                validate_sharding_integrity = True
                save_strategy = (checkpointing_context or {}).get('save_strategy',
                                                                get_default_save_sharded_strategy(args.dist_ckpt_format))
                if args.ckpt_assume_constant_structure and args.dist_ckpt_format == 'torch_dist':
                    save_strategy.use_cached_ckpt_structure = args.ckpt_assume_constant_structure
                if args.ckpt_fully_parallel_save:
                    if checkpointing_context is not None and 'save_strategy' in checkpointing_context:
                        # Already saved once before - don't need to rerun sharding validation
                        validate_sharding_integrity = not args.ckpt_assume_constant_structure
                    else:
                        save_strategy = FullyParallelSaveStrategyWrapper(save_strategy, mpu.get_data_parallel_group(with_context_parallel=True),
                                                                        args.ckpt_assume_constant_structure)
                # Store save strategy for future checkpoint saves
                if checkpointing_context is not None:
                    checkpointing_context['save_strategy'] = save_strategy
                end_ckpt = time()
                if not torch.distributed.is_initialized():
                    logger.debug(f"takes {end_misc - start_misc} to finalize ckpt save ")
                else:
                    logger.debug(f"rank: {torch.distributed.get_rank()}, takes {end_ckpt - start_ckpt} to prepare state dict for ckpt ")
                async_save_request = dist_checkpointing.save(state_dict, checkpoint_name, save_strategy,
                                                            async_sharded_save=args.async_save)

                # [ModelOpt]: save sharded modelopt_state
                if has_nvidia_modelopt:
                    save_sharded_modelopt_state(model, checkpoint_name, (args.dist_ckpt_format, 1))
            else:
                # [ModelOpt]: Inject modelopt_state into state_dict
                if has_nvidia_modelopt:
                    save_modelopt_state(model, state_dict)

                # Save.
                ensure_directory_exists(checkpoint_name)
                torch.save(state_dict, checkpoint_name)
    
    if args.deepspeed:
        #megatron model uses state_dict_for_save_checkpointing instead of the standard state_dict
        #state_dict is used by deepspeed for module saving so it needs to point to the right function
        if args.no_pipeline_parallel:
            original_state_dict = model[0].module.state_dict
            def state_dict_for_save_checkpoint_deepspeed(destination=None, prefix='', keep_vars=False):
                return model[0].module.state_dict_for_save_checkpoint(prefix=prefix, keep_vars=keep_vars)
            model[0].module.state_dict = state_dict_for_save_checkpoint_deepspeed

        # Saving is a collective communication
        checkpoint_name = get_checkpoint_name(args.save, iteration)

        # Trim off the filename and mp_rank_* directory.
        for _ in range(3):
            checkpoint_name = os.path.dirname(checkpoint_name)
        model[0].save_checkpoint(checkpoint_name, client_state=state_dict)

        if args.no_pipeline_parallel:
            model[0].module.state_dict = original_state_dict

    if not args.deepspeed:
        start_misc = time()
        if not args.async_save:
            assert async_save_request is None
            # Wait so everyone is done (necessary)
            if torch.distributed.is_initialized():
                torch.distributed.barrier()

        # And update the latest iteration
        if not torch.distributed.is_initialized() \
            or torch.distributed.get_rank() == 0 \
            or (getattr(args, 'data_cache_local', None) and torch.distributed.get_rank() % torch.cuda.device_count() == 0):
            tracker_filename = get_checkpoint_tracker_filename(args.save)

            def iter_finalize_fn():
                with open(tracker_filename, 'w') as f:
                    f.write(str(iteration))
                print_rank_0('  successfully saved checkpoint from iteration {:7d} to {}'
                            .format(iteration, args.save))
                if args.log_progress and args.async_save:
                    append_to_progress_log(f'Saved async checkpoint\tIteration: {iteration}',
                                        barrier=False)

            if args.async_save:
                assert async_save_request is not None
                async_save_request.add_finalize_fn(iter_finalize_fn)
            else:
                iter_finalize_fn()

        # Additional callback for one_logger (last rank)
        if not torch.distributed.is_initialized() \
        or is_last_rank():
            def onelogger_finalize_fn():
                on_save_checkpoint_success(productive_metrics, args.async_save)
            if args.async_save:
                assert async_save_request is not None
                async_save_request.add_finalize_fn(onelogger_finalize_fn)
            else:
                onelogger_finalize_fn()

        if args.async_save:
            schedule_async_save(async_save_request)
            print_rank_0('  scheduled an async checkpoint save at iteration {:7d} to {}' \
                        .format(iteration, args.save))

        # Wait so everyone is done (not necessary)
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        end_misc = time()
        if not torch.distributed.is_initialized():
            logger.debug(f"takes {end_misc - start_misc} to finalize ckpt save ")
        else:
            logger.debug(f"rank: {torch.distributed.get_rank()}, takes {end_misc - start_misc} to finalize ckpt save ")
    else:
        # Wait so everyone is done (necessary)
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        print_rank_0('  successfully saved checkpoint at iteration {:7d} to {}' \
                    .format(iteration, args.save))

        # And update the latest iteration
        if not torch.distributed.is_initialized() \
            or torch.distributed.get_rank() == 0 \
            or (getattr(args, 'data_cache_local', None) and torch.distributed.get_rank() % torch.cuda.device_count() == 0):
            tracker_filename = get_checkpoint_tracker_filename(args.save)
            with open(tracker_filename, 'w') as f:
                f.write(str(iteration))

        # Wait so everyone is done (not necessary)
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

def generate_state_dict(args, model, optimizer, opt_param_scheduler,
                        rng_state, use_dist_ckpt=False, iteration=None,
                        optim_sd_kwargs=None):
    # Arguments, iteration, and model.
    state_dict = {}
    state_dict['args'] = args
    state_dict['checkpoint_version'] = 3.0
    if iteration is not None:
        state_dict['iteration'] = iteration
    state_dict['tokens'] = args.consumed_train_tokens

    if args.deepspeed:
        state_dict[UNIVERSAL_CHECKPOINT_INFO] = _universal_checkpoint_info(model)

    # DeepSpeed saves the model/optimizer/scheduler
    if not args.deepspeed:
        if len(model) == 1:
            state_dict['model'] = (model[0].sharded_state_dict()
                                if use_dist_ckpt else
                                model[0].state_dict_for_save_checkpoint())
        else:
            for i in range(len(model)):
                mpu.set_virtual_pipeline_model_parallel_rank(i)
                state_dict['model%d' % i] = (
                    model[i].sharded_state_dict()
                    if use_dist_ckpt else
                    model[i].state_dict_for_save_checkpoint())
        # Optimizer stuff.
        if not args.no_save_optim:
            if optimizer is not None:
                state_dict['optimizer'] = (optimizer.sharded_state_dict(state_dict, **(optim_sd_kwargs or {}))
                                        if use_dist_ckpt else
                                        optimizer.state_dict())
            if opt_param_scheduler is not None:
                state_dict['opt_param_scheduler'] = \
                    opt_param_scheduler.state_dict()
    # RNG states.
    if not args.no_save_rng:
        state_dict["rng_state"] = rng_state
    return state_dict

# Not used! num_key_value_heads can be replaced with megatron-lm num_query_groups.
def load_args_from_checkpoint_wrapper(load_args_from_checkpoint):
    @wraps(load_args_from_checkpoint)
    def wrapper(args, load_arg='load', exit_on_missing_checkpoint=False):
        args, checkpoint_args = load_args_from_checkpoint(args, load_arg, exit_on_missing_checkpoint)

        def _set_arg(arg_name, old_arg_name=None, force=False):
            if not force and getattr(args, arg_name, None) is not None:
                return

            if old_arg_name is not None:
                checkpoint_value = getattr(checkpoint_args, old_arg_name, None)
            else:
                checkpoint_value = getattr(checkpoint_args, arg_name, None)

            if checkpoint_value is not None:
                print_rank_0(f"Setting {arg_name} to {checkpoint_value} from checkpoint")
                setattr(args, arg_name, checkpoint_value)
            else:
                print_rank_0(f"Checkpoint did not provide arguments {arg_name}")

        _set_arg('num_key_value_heads')

        return args, checkpoint_args
    return wrapper


def load_checkpoint(model, optimizer, opt_param_scheduler, load_arg='load', strict=True, load_only_weights=False):
    """Load a model checkpoint and return the iteration.
    strict (bool): whether to strictly enforce that the keys in
        :attr:`state_dict` of the checkpoint match the names of
        parameters and buffers in model.
    """
    args = get_args()
    load_dir = getattr(args, load_arg)

    # Finetuning directories
    pretrained_dir = getattr(args,'pretrained_checkpoint', None)
    if pretrained_dir is not None and not checkpoint_exists(load_dir):
        print_rank_0(f'Checkpoint file not found in load directory {load_dir} attempting to finetune with checkpoint in {pretrained_dir}')
        load_dir = pretrained_dir
        if not checkpoint_exists(load_dir):
            raise FileNotFoundError("No checkpoint found in load directory or pretrained directory")
        args.finetune = True

    if args.deepspeed:
        if args.finetune:
            loaded_dir, state_dict = model[0].load_checkpoint(load_dir,
                load_module_strict=strict, load_optimizer_states=False,
                load_lr_scheduler_states=False, load_module_only=True,
                tag=args.load_tag)
        else:
            loaded_dir, state_dict = model[0].load_checkpoint(load_dir,
                load_module_strict=strict, tag=args.load_tag)
        if loaded_dir is None:
            print_rank_0('WARNING: could not find the metadata file {} '.format(
                load_dir))
            print_rank_0('    will not load any checkpoints and will start from '
                        'random')
            return 0, 0
        release = False        
    else:
        model = unwrap_model(model)

        load_kwargs = {}
        is_dist_ckpt = False
        if args.auto_detect_ckpt_format or args.use_dist_ckpt:
            state_dict, checkpoint_name, release = _load_base_checkpoint(load_dir, rank0=True, exit_on_missing_checkpoint=args.exit_on_missing_checkpoint)
            is_dist_ckpt = dist_checkpointing.check_is_distributed_checkpoint(checkpoint_name)
            if is_dist_ckpt:
                ckpt_tp_pp = (state_dict['args'].tensor_model_parallel_size, state_dict['args'].pipeline_model_parallel_size)
                run_tp_pp = (mpu.get_tensor_model_parallel_world_size(), mpu.get_pipeline_model_parallel_world_size())
                mismatch_msg = "(TP, PP) mismatch after resume ({} vs {} from checkpoint)".format(ckpt_tp_pp, run_tp_pp)

                # Determine if RNG state will be loaded
                if (ckpt_tp_pp == run_tp_pp and not release and not args.finetune and not args.no_load_rng
                        and not getattr(state_dict['args'], 'no_save_rng', False)):
                    gen_sd_rng_state = get_rng_state(True)  # we can load the rng state
                else:
                    gen_sd_rng_state = None
                    if ckpt_tp_pp != run_tp_pp:
                        print_rank_0("{}: RNG state will be ignored".format(mismatch_msg))

                optim_sd_kwargs = dict(is_loading=True)
                # Determine if optimizer state will be loaded
                if (not release and not args.finetune and not args.no_load_optim
                        and not getattr(state_dict['args'], 'no_save_optim', False)):
                    gen_sd_optim = optimizer
                    gen_sd_opt_param_scheduler = opt_param_scheduler

                    if args.use_distributed_optimizer:
                        optim_sd_kwargs['sharding_type'] = ('fully_sharded_model_space'
                                                            if getattr(state_dict['args'], 'ckpt_fully_parallel_save', False)
                                                            else 'dp_zero_gather_scatter')
                        # This is for backwards-compatibility. Can be removed once 'fully_sharded_bucket_space' loading is removed
                        for maybe_dist_opt_optim_state in (state_dict['optimizer'], *state_dict['optimizer'].values()):
                            if 'param_state_sharding_type' in maybe_dist_opt_optim_state:
                                if maybe_dist_opt_optim_state['param_state_sharding_type'] == 'fully_sharded_bucket_space':
                                    print_rank_0('Detected deprecated `fully_sharded_bucket_space` DistributedOptimizer checkpoint format')
                                    optim_sd_kwargs['sharding_type'] = maybe_dist_opt_optim_state['param_state_sharding_type']
                                break

                        if ckpt_tp_pp != run_tp_pp and optim_sd_kwargs['sharding_type'] != 'fully_sharded_model_space':
                            raise RuntimeError(f"{mismatch_msg}: not supported for DistributedOptimizer with sharding type {optim_sd_kwargs['sharding_type']}."
                                            f" Please use `--ckpt-fully-parallel-save` flag during checkpoint saving.")
                else:
                    gen_sd_optim = None
                    gen_sd_opt_param_scheduler = None
                load_kwargs['sharded_state_dict'] = generate_state_dict(args, model, gen_sd_optim, gen_sd_opt_param_scheduler,
                                                                        gen_sd_rng_state, True, optim_sd_kwargs=optim_sd_kwargs)
                load_kwargs['exit_on_missing_checkpoint'] = args.exit_on_missing_checkpoint

        state_dict, checkpoint_name, release = _load_base_checkpoint(load_dir, rank0=False, **load_kwargs)

        # Checkpoint not loaded.
        if state_dict is None:
            # Iteration and num_floating_point_operations_so_far default to 0.
            return 0, 0
    checkpoint_name = get_checkpoint_name(load_dir, state_dict['iteration'], release)

    # Set checkpoint version.
    set_checkpoint_version(state_dict.get('checkpoint_version', 0))

    # Set iteration.
    if args.finetune or release or load_only_weights:
        iteration = 0
        # Make DeepSpeed engine aware of this reset of iteration
        model[0].global_steps = 0
    else:
        try:
            iteration = state_dict['iteration']
            if 'tokens' in state_dict:
                args.consumed_train_tokens = state_dict['tokens']
        except KeyError:
            try:  # Backward compatible with older checkpoints
                iteration = state_dict['total_iters']
            except KeyError:
                print_rank_0('A metadata file exists but unable to load '
                             'iteration from checkpoint {}, exiting'.format(checkpoint_name))
                sys.exit()
    num_floating_point_operations_so_far = state_dict.get('num_floating_point_operations_so_far', 0)

    # Check arguments.
    if not load_only_weights:
        assert args.consumed_train_samples == 0
        assert args.consumed_valid_samples == 0
        if 'args' in state_dict and not args.finetune:
            checkpoint_args = state_dict['args']
            check_checkpoint_args(checkpoint_args)
            args.consumed_train_samples = getattr(checkpoint_args,
                                                'consumed_train_samples', 0)
            update_num_microbatches(consumed_samples=args.consumed_train_samples)
            args.consumed_valid_samples = getattr(checkpoint_args,
                                                'consumed_valid_samples', 0)
        else:
            print_rank_0('could not find arguments in the checkpoint ...')

    # [ModelOpt]: loading modelopt_state (sharded or not)
    if has_nvidia_modelopt:
        if args.use_dist_ckpt:
            restore_sharded_modelopt_state(model, checkpoint_name)
        else:
            restore_modelopt_state(model, state_dict)

    # Model.
    if not args.deepspeed:
        strict = False if args.retro_add_retriever else strict
        if len(model) == 1:
            model[0].load_state_dict(state_dict['model'], strict=strict)
        else:
            for i in range(len(model)):
                mpu.set_virtual_pipeline_model_parallel_rank(i)
                model[i].load_state_dict(state_dict['model%d' % i], strict=strict)

    # Fix up query/key/value matrix ordering if needed.
    checkpoint_version = get_checkpoint_version()
    print_rank_0(f' checkpoint version {checkpoint_version}')
    fix_query_key_value_ordering(model, checkpoint_version)

    # Optimizer.
    if not args.deepspeed:
        if not release and not args.finetune and not args.no_load_optim:
            try:
                # Load state dict.
                if optimizer is not None:
                    optimizer.load_state_dict(state_dict['optimizer'])

                # Load distributed optimizer's custom parameter state.
                # For distributed checkpoint it's already loaded in load_state_dict above
                if args.use_distributed_optimizer and not is_dist_ckpt:
                    tracker_filename = get_checkpoint_tracker_filename(load_dir)
                    iteration, release = read_metadata(tracker_filename)
                    model_checkpoint_name = \
                        get_checkpoint_name(load_dir, iteration, release)
                    optim_checkpoint_name = \
                        get_distributed_optimizer_checkpoint_name(
                            model_checkpoint_name)
                    optimizer.load_parameter_state(optim_checkpoint_name)

                # Load scheduler.
                if opt_param_scheduler is not None:
                    if 'lr_scheduler' in state_dict: # backward compatbility
                        opt_param_scheduler.load_state_dict(state_dict['lr_scheduler'])
                    else:
                        opt_param_scheduler.load_state_dict(state_dict['opt_param_scheduler'])
            except KeyError:
                print_rank_0('Unable to load optimizer from checkpoint {}. '
                            'Specify --no-load-optim or --finetune to prevent '
                            'attempting to load the optimizer state, '
                            'exiting ...'.format(checkpoint_name))
                sys.exit()
        else:
            if (args.fp16 or args.bf16) and optimizer is not None:
                optimizer.reload_model_params()

    # rng states.
    if not release and not args.finetune and not args.no_load_rng:
        try:
            if 'rng_state' in state_dict:
                # access rng_state for data parallel rank
                if args.data_parallel_random_init:
                    rng_state = state_dict['rng_state'][mpu.get_data_parallel_rank()]
                else:
                    rng_state = state_dict['rng_state'][0]
                random.setstate(rng_state['random_rng_state'])
                np.random.set_state(rng_state['np_rng_state'])
                torch.set_rng_state(rng_state['torch_rng_state'])
                torch.cuda.set_rng_state(rng_state['cuda_rng_state'])
                # Check for empty states array
                if not rng_state['rng_tracker_states']:
                    raise KeyError
                tensor_parallel.get_cuda_rng_tracker().set_states(
                    rng_state['rng_tracker_states'])
            else:  # backward compatability
                random.setstate(state_dict['random_rng_state'])
                np.random.set_state(state_dict['np_rng_state'])
                torch.set_rng_state(state_dict['torch_rng_state'])
                torch.cuda.set_rng_state(state_dict['cuda_rng_state'])
                # Check for empty states array
                if not state_dict['rng_tracker_states']:
                    raise KeyError
                tensor_parallel.get_cuda_rng_tracker().set_states(
                    state_dict['rng_tracker_states'])
        except KeyError:
            print_rank_0('Unable to load rng state from checkpoint {}. '
                         'Specify --no-load-rng or --finetune to prevent '
                         'attempting to load the rng state, '
                         'exiting ...'.format(checkpoint_name))
            sys.exit()

        if args.universal_checkpoint:
            # TLDR: unique rng is needed for dropout to be really random on TP ranks
            #
            # Each tp-rank stores its model-parallel-rng states info.
            # This is required to e.g. have different dropout patterns on different tp ranks that operate on
            # slices of attention_probs tensor.
            #
            # When loading from universal checkpoint, we use mp_rank_<mp>_model_states.pt checkpoint files
            # to restore the model-parallel-rng (<mp> is {tp-rank, pp-rank} combination).
            # However, if the loaded checkpoint mp configuration does not match the current mp configuration,
            # we can not use it to restore model-parallel-rng info.
            #
            # In the case of mp configuration change, we reconfigure the model-parallel-rng states s.t. each
            # tp-rank will have a unique state. In order to ensure that subsequent loads from universal will
            # not cause the model-parallel-rng states to be repeated, we add the iteration number to the base seed.
            ckp_args = state_dict['args']
            if ((args.tensor_model_parallel_size != ckp_args.tensor_model_parallel_size)
                    or (args.pipeline_model_parallel_size != ckp_args.pipeline_model_parallel_size)):
                print_rank_0(' loading universal checkpoint with modified mp configuration '
                             '-> reconfigure tp seed')
                tensor_parallel.model_parallel_reconfigure_tp_seed(args.seed + iteration)

    # Some utilities want to load a checkpoint without distributed being initialized
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    print_rank_0(f'  successfully loaded checkpoint from {load_dir} '
                 f'[ t {mpu.get_tensor_model_parallel_rank()}, '
                 f'p {mpu.get_pipeline_model_parallel_rank()} ] '
                 f'at iteration {iteration}')

    return iteration, num_floating_point_operations_so_far

def _universal_checkpoint_info(model):
    args = get_args()
    tokenizer = get_tokenizer()
    info = dict()
    info[UNIVERSAL_CHECKPOINT_VERSION_KEY] = UNIVERSAL_CHECKPOINT_VERSION_VALUE
    info[ORIGINAL_VOCAB_SIZE] = tokenizer.vocab_size
    info[PADDED_VOCAB_SIZE] = args.padded_vocab_size
    info.update(model[0].universal_checkpoint_info())
    return info
