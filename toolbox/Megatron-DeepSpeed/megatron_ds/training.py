# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

"""Pretrain utilities."""

import gc
from datetime import datetime
import math
import logging
import sys
from .log_handler import CustomHandler
# Make default logging level INFO, but filter out all log messages not from MCore.
logging.basicConfig(handlers=[CustomHandler()], level=logging.INFO)
from .theoretical_memory_usage import report_theoretical_memory
import time
import json
import os
# The earliest we can measure the start time.
_TRAIN_START_TIME = time.time()
import torch
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP

from megatron_ds import get_args
from megatron_ds import get_signal_handler
from megatron_ds import get_timers
from megatron_ds import get_tensorboard_writer
from megatron_ds import get_wandb_writer
from megatron_ds import get_current_global_batch_size
from megatron_ds import get_num_microbatches
from megatron_ds import is_last_rank
from megatron_ds import update_num_microbatches
from megatron_ds.core import mpu, tensor_parallel
from megatron_ds.core.utils import get_model_config
from megatron_ds import print_rank_0
from megatron_ds import print_rank_last
from megatron_ds.checkpointing import load_checkpoint
from megatron_ds.checkpointing import save_checkpoint
from megatron_ds.model import Float16Module
from megatron_ds.model import GPTModel
from megatron_ds.core.distributed import DistributedDataParallel as DDP
from megatron_ds.core.distributed import finalize_model_grads
from megatron_ds.core.enums import ModelType
from megatron_ds.optimizer import get_megatron_optimizer
from megatron_ds.initialize import initialize_megatron
from megatron_ds.initialize import write_args_to_tensorboard
from megatron_ds.initialize import set_jit_fusion_options
from megatron_ds.optimizer_param_scheduler import OptimizerParamScheduler
from megatron_ds.model import DistributedDataParallel as LocalDDP
from megatron_ds.utils import check_adlr_autoresume_termination
from megatron_ds.utils import unwrap_model
from megatron_ds.data.data_samplers import build_pretraining_data_loader
from megatron_ds.utils import calc_params_l2_norm
from megatron_ds.core.pipeline_parallel import get_forward_backward_func
from megatron_ds.utils import report_memory, throughput_calculator, checkpoint_throughput_calculator, update_rotary_pos_emb
# from megatron.model.vision.knn_monitor import compute_feature_bank
from megatron_ds.arguments import core_transformer_config_from_args

import deepspeed
from deepspeed.accelerator import get_accelerator
from deepspeed.compression.compress import init_compression, redundancy_clean
from deepspeed.runtime.data_pipeline.data_routing.helper import convert_to_random_ltd
from megatron_ds.model.transformer import ParallelTransformerLayer

from deepspeed import comm as dist

try:
    import wandb
except (ImportError, ModuleNotFoundError):
    wandb = None


def execCmd(cmd):
    r = os.popen(cmd)
    text = r.read()
    r.close()
    return text

def print_datetime(string):
    """Note that this call will sync across all ranks."""
    torch.distributed.barrier()
    time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print_rank_0('[' + string + '] datetime: {} '.format(time_str))

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
    

def num_floating_point_operations(args, batch_size):
    if not args.group_query_attention:
        args.num_query_groups = args.num_attention_heads
    return (
        60
        * batch_size
        * args.seq_length
        * args.num_layers
        * args.hidden_size
        * args.hidden_size
        * (
            1
            + (args.num_query_groups / (5 * args.num_attention_heads))
            + (args.seq_length / (5 * args.hidden_size))
            + (args.padded_vocab_size / (10 * args.num_layers * args.hidden_size))
        )
    )


def pretrain(train_valid_test_dataset_provider,
             model_provider,
             model_type,
             forward_step_func,
             process_non_loss_data_func=None,
             extra_args_provider=None,
             args_defaults={},
             data_post_process=None,
             external_args={}):
    """Main training program.

    This function will run the followings in the order provided:
        1) initialize Megatron.
        2) setup model, optimizer and lr schedule using the model_provider.
        3) call train_val_test_data_provider to get train/val/test datasets.
        4) train the modle using the forward_step_func.

    Arguments:
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
    """

    # Initalize and get arguments, timers, and Tensorboard writer.
    initialize_megatron(extra_args_provider=extra_args_provider,
                        args_defaults=args_defaults, external_args=external_args)
    # Set pytorch JIT layer fusion options and warmup JIT functions.
    if get_accelerator().device_name() == 'cuda':
        set_jit_fusion_options()

    # Adjust the startup time so it reflects the largest value.
    # This will be closer to what scheduler will see (outside of
    # image ... launches.
    global _TRAIN_START_TIME
    start_time_tensor = torch.cuda.DoubleTensor([_TRAIN_START_TIME])
    torch.distributed.all_reduce(start_time_tensor,
                                 op=torch.distributed.ReduceOp.MIN)
    _TRAIN_START_TIME = start_time_tensor.item()
    print_rank_0('time to initialize megatron (seconds): {:.3f}'.format(
        time.time() - _TRAIN_START_TIME))
    print_datetime('after megatron is initialized')

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

    # Model, optimizer, and learning rate.
    timers('model-and-optimizer-setup', log_level=0).start(barrier=True)
    model, optimizer, opt_param_scheduler = setup_model_and_optimizer(
        model_provider, model_type, teacher=False, data_post_process=data_post_process,
        build_train_valid_test_datasets_provider=train_valid_test_dataset_provider)
    timers('model-and-optimizer-setup').stop()
    print_datetime('after model, optimizer, and learning rate '
                   'scheduler are built')
    if args.deepspeed:
        config = core_transformer_config_from_args(args)
    else:
        config = get_model_config(model[0])

    # Data stuff.
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

    # args.teacher_model is used as global variable to pass the teacher model
    # for knowledge distillation. Users do not need to set it in the command
    # line to use kd, but users do need to provide teacher model configurations
    # like args.num_layers_teacher as described in setup_teacher_model()
    args.teacher_model = None
    if args.mos or args.kd: # Set up teacher model
        args.teacher_model = setup_teacher_model(args, model_provider)

    # Print setup timing.
    print_rank_0('done with setup ...')
    timers.log(['model-and-optimizer-setup',
                'train/valid/test-data-iterators-setup'], barrier=True)

    if not args.skip_train:
        print_rank_0('training ...')

        if args.dataloader_type == 'cyclic' and args.retro_add_retriever:
            args.train_iters = args.retro_cyclic_train_iters
            print_rank_0("retro cyclic train iters : %d" % args.train_iters)

        iteration = 0
        if args.do_train and args.train_iters > 0:
            iteration = train(forward_step_func,
                              model, optimizer, opt_param_scheduler,
                              train_data_iterator, valid_data_iterator,
                              process_non_loss_data_func, config)

        print_datetime('after training is done')
        # Clean the model
        if args.compression_training:
            model = [redundancy_clean(model[0], args.deepspeed_config_dict, mpu)]

        if args.save and iteration != 0:
            save_checkpoint(iteration, model, optimizer, opt_param_scheduler)
    else:
        print_rank_0('skipping training (--skip-train is on) ...')

        iteration = args.iteration

    if args.do_valid:
        prefix = f'iteration {iteration} on validation set'
        evaluate_and_print_results(prefix, forward_step_func,
                                   valid_data_iterator, model,
                                   iteration, process_non_loss_data_func, config,
                                   verbose=True, write_to_tensorboard=not args.skip_train)

    if args.do_test:
        prefix = f'iteration {iteration} on test set'
        evaluate_and_print_results(prefix, forward_step_func,
                                   test_data_iterator, model,
                                   iteration, process_non_loss_data_func, config,
                                   verbose=True, write_to_tensorboard=not args.skip_train)


def update_train_iters(args):

    # For iteration-based training, we don't need to do anything
    if args.train_iters:
        return

    # Constant batch size with sample-based training.
    if args.rampup_batch_size is None:
        args.train_iters = args.train_samples // args.global_batch_size

    else:
        # Sample based training with rampup batch size.
        iterations = 0
        consumed_samples = 0
        # Rampup phase.
        while consumed_samples <= int(args.rampup_batch_size[2]):
            update_num_microbatches(consumed_samples, consistency_check=False)
            consumed_samples += get_current_global_batch_size()
            iterations += 1
        # Reset
        update_num_microbatches(0, consistency_check=False)
        # Constant phase
        # Note that we throw away any partial last batch.
        iterations += (args.train_samples - consumed_samples) // \
                      args.global_batch_size
        args.train_iters = iterations

    print_rank_0('setting training iterations to {}'.format(args.train_iters))


def setup_teacher_model(args, model_provider):        
    
    print_rank_0('***>>>>> Student model checkpoint iteration:{}'.format(args.iteration))
    iteration_stuent = args.iteration
    num_layers_student = args.num_layers
    num_experts_student = args.num_experts
    hidden_size_student = args.hidden_size
    num_attention_heads_student = args.num_attention_heads
    load_student = args.load

    print_rank_0('***>>>>> Setting up the teacher model')

    args.num_layers = args.num_layers_teacher
    args.num_experts = args.num_experts_teacher
    args.hidden_size = args.hidden_size_teacher
    args.num_attention_heads = args.num_attention_heads_teacher
    args.load = args.load_teacher
    teacher_model, _, _ = load_model_weights_only(model_provider)
    print_rank_0('***>>>>> Teacher model:{}'.format(teacher_model))

    args.num_layers = num_layers_student
    args.num_experts = num_experts_student
    args.hidden_size = hidden_size_student
    args.num_attention_heads = num_attention_heads_student
    args.load = load_student
    args.iteration = iteration_stuent

    return teacher_model

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
                assert args.pipeline_model_parallel_split_rank is not None, \
                    "Split rank needs to be specified for model with both encoder and decoder"
                rank = mpu.get_pipeline_model_parallel_rank()
                split_rank = args.pipeline_model_parallel_split_rank
                world_size = mpu.get_pipeline_model_parallel_world_size()
                pre_process = rank == 0 or rank == split_rank
                post_process = (rank == (split_rank - 1)) or (
                        rank == (world_size - 1))
                add_encoder = mpu.is_pipeline_stage_before_split()
                add_decoder = mpu.is_pipeline_stage_after_split()
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

    # Disallow training and inference with Transformer Engine
    # for non-GPT models
    args.allow_transformer_engine = all([type(m) == GPTModel for m in model])
    # assert args.allow_transformer_engine or args.transformer_impl == 'local', \
    #     'Transformer Engine is only approved for GPT models'

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

    if wrap_with_ddp:
        config = get_model_config(model[0])
        model = [DDP(config,
                     model_chunk,
                     data_parallel_group=mpu.get_data_parallel_group(with_context_parallel=True),
                     accumulate_allreduce_grads_in_fp32=args.accumulate_allreduce_grads_in_fp32,
                     overlap_grad_reduce=args.overlap_grad_reduce,
                     use_distributed_optimizer=args.use_distributed_optimizer,
                     # Turn off bucketing for model_chunk 2 onwards, since communication for these
                     # model chunks is overlapped with compute anyway.
                     disable_bucketing=(model_chunk_idx > 0))
                 for (model_chunk_idx, model_chunk) in enumerate(model)]

        # Broadcast params from data parallel src rank to other data parallel ranks.
        if args.data_parallel_random_init:
            for model_module in model:
                model_module.broadcast_params()

    return model


def get_optimizer_param_scheduler(optimizer):
    """Build the learning rate scheduler."""
    args = get_args()

    # Iteration-based training.
    if args.train_iters:
        if args.lr_decay_iters is None:
            args.lr_decay_iters = args.train_iters
        lr_decay_steps = args.lr_decay_iters * args.global_batch_size
        wd_incr_steps = args.train_iters * args.global_batch_size
        if args.lr_warmup_fraction is not None:
            lr_warmup_steps = args.lr_warmup_fraction * lr_decay_steps
        else:
            lr_warmup_steps = args.lr_warmup_iters * args.global_batch_size
    # Sample-based training.
    elif args.train_samples:
        # We need to set training iters for later use. Technically
        # we need to adjust the training samples too (due to last
        # batch being incomplete) but we leave it as is for now.
        update_train_iters(args)
        if args.lr_decay_samples is None:
            args.lr_decay_samples = args.train_samples
        lr_decay_steps = args.lr_decay_samples
        wd_incr_steps = args.train_samples
        if args.lr_warmup_fraction is not None:
            lr_warmup_steps = args.lr_warmup_fraction * lr_decay_steps
        else:
            lr_warmup_steps = args.lr_warmup_samples
    else:
        raise Exception(
            'either train-iters or train-samples should be provided.')

    opt_param_scheduler = OptimizerParamScheduler(
        optimizer,
        init_lr=args.lr_warmup_init,
        max_lr=args.lr,
        min_lr=args.min_lr,
        lr_warmup_steps=lr_warmup_steps,
        lr_decay_steps=lr_decay_steps,
        lr_decay_style=args.lr_decay_style,
        start_wd=args.start_weight_decay,
        end_wd=args.end_weight_decay,
        wd_incr_steps=wd_incr_steps,
        wd_incr_style=args.weight_decay_incr_style,
        use_checkpoint_opt_param_scheduler=args.use_checkpoint_opt_param_scheduler,
        override_opt_param_scheduler=args.override_opt_param_scheduler)

    return opt_param_scheduler

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
        iteration = load_checkpoint(model, optimizer, lr_scheduler, strict=True, load_only_weights=True)

    print_datetime('after load checkpoint weights')

    return model, optimizer, lr_scheduler


def setup_model_and_optimizer(model_provider_func,
                              model_type,
                              no_wd_decay_cond=None,
                              scale_lr_cond=None,
                              lr_mult=1.0,
                              teacher=False,
                              data_post_process=None,
                              build_train_valid_test_datasets_provider=None):
    """Setup model and optimizer."""
    args = get_args()

    model = get_model(model_provider_func, model_type)
    # unwrapped_model = unwrap_model(model)

    # initialize the compression here
    student_global_steps = 0
    if args.kd or args.mos:
        model, _, _, _ = deepspeed.initialize(
                model=model[0],
                args=args,
                mpu=mpu if args.no_pipeline_parallel else None,
                config=args.deepspeed_config_dict,
            )
        model = [model]
        if args.load is not None:
            args.iteration = load_checkpoint(model, None, None, strict=False)
        else:
            args.iteration = 0
        student_global_steps = model[0].global_steps
        print_rank_0('***>>>>> Student model, global step:{}'.format(student_global_steps))

    if args.compression_training:
        model, _, _, _ = deepspeed.initialize(
            model=model[0],
            args=args,
            mpu=mpu if args.no_pipeline_parallel else None,
            config=args.deepspeed_config_dict,
        )
        model = [model]
        model = [init_compression(model[0].module, args.deepspeed_config_dict, mpu)]

    unwrapped_model = unwrap_model(model,
                                   (torchDDP, LocalDDP, DDP, Float16Module))

    if args.inference:
        optimizer = None
        opt_param_scheduler = None
    else:
        if teacher:
            optimizer = None
        else:
            optimizer = get_megatron_optimizer(model, no_wd_decay_cond,
                                               scale_lr_cond, lr_mult)
        # opt_param_scheduler is the old lr_scheduler plus weight decay scheduling
        opt_param_scheduler = get_optimizer_param_scheduler(optimizer)

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

    # Compression has its own checkpoint loading path (e.g, loading both teacher and student models). So if compression is enabled, we skip the following checkpoint loading.
    no_post_init_checkpoint_loading = args.kd or args.mos
    if not no_post_init_checkpoint_loading:
        if args.load is not None:
            timers = get_timers()
            timers('load-checkpoint', log_level=0).start(barrier=True)
            args.iteration = load_checkpoint(model, optimizer, opt_param_scheduler)
            timers('load-checkpoint').stop(barrier=True)
            timers.log(['load-checkpoint'])
        else:
            args.iteration = 0
    else:
        model[0].global_steps = student_global_steps

    # We only support local DDP with multiple micro-batches.
    if len(model) > 1 or mpu.get_pipeline_model_parallel_world_size() > 1:
        assert args.DDP_impl == 'local'

    # get model without FP16 and/or TorchDDP wrappers
    if args.iteration == 0 and len(unwrapped_model) == 1 \
        and hasattr(unwrapped_model[0], 'init_state_dict_from_bert'):
        print_rank_0("Initializing ICT from pretrained BERT model")
        unwrapped_model[0].init_state_dict_from_bert()
        if args.fp16:
            optimizer.reload_model_params()

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
        grad_norm = model[0].get_global_grad_norm()
        return {'lm loss' : loss}, skipped_iter, grad_norm, num_zeros_in_grad

    # Set grad to zero.
    for model_chunk in model:
        # If using distributed optimizer, don't zero buffer here; zeroing of buffer is
        # handled automatically by the optimizer after all-gathers finish.
        # Otherwise, zero the buffer.
        model_chunk.zero_grad_buffer(zero_buffer=(not args.use_distributed_optimizer))
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

    # Empty unused memory.
    if args.empty_unused_memory_level >= 1:
        torch.cuda.empty_cache()

    # Vision gradients.
    if args.vision_pretraining and args.vision_pretraining_type == "dino":
        unwrapped_model = unwrap_model(model[0])
        unwrapped_model.cancel_gradients_last_layer(args.curr_iteration)

    # Update parameters.
    timers('optimizer', log_level=1).start(barrier=args.barrier_with_L1_time)
    update_successful, grad_norm, num_zeros_in_grad = optimizer.step(args, timers)
    timers('optimizer').stop()

    # Vision momentum.
    if args.vision_pretraining and args.vision_pretraining_type == "dino":
        unwrapped_model = unwrap_model(model[0])
        unwrapped_model.update_momentum(args.curr_iteration)

    # Update learning rate.
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
        for key in losses_reduced[0]:
            losses_reduced_for_key = [x[key] for x in losses_reduced]
            loss_reduced[key] = sum(losses_reduced_for_key) / len(losses_reduced_for_key)
        return loss_reduced, skipped_iter, grad_norm, num_zeros_in_grad
    return {}, skipped_iter, grad_norm, num_zeros_in_grad


def training_log(loss_dict, total_loss_dict, learning_rate, iteration,
                 loss_scale, report_memory_flag, skipped_iter,
                 grad_norm, params_norm, num_zeros_in_grad,
                 model=None, optimizer=None):
    """Log training information such as losses, timing, ...."""
    args = get_args()
    timers = get_timers()
    writer = get_tensorboard_writer()
    wandb_writer = get_wandb_writer()

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
                key, torch.cuda.FloatTensor([0.0])) + loss_dict[key]
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

    total_iterations = total_loss_dict[advanced_iters_key] + \
                       total_loss_dict[skipped_iters_key]

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
        if args.log_learning_rate_to_tensorboard:
            writer.add_scalar('learning-rate', learning_rate, iteration)
            writer.add_scalar('learning-rate vs samples', learning_rate,
                              args.consumed_train_samples)
            if wandb_writer:
                wandb_writer.log({'learning-rate': learning_rate}, iteration)
        if args.log_batch_size_to_tensorboard:
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
                "mem-allocated-count",
                mem_stats["allocation.all.current"],
                iteration,
            )

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
        if writer:
            if args.log_timers_to_tensorboard:
                writer.add_scalar('iteration-time/iteration-time',
                                  elapsed_time_per_iteration, iteration)
            if wandb_writer:
                wandb_writer.log({'iteration-time': elapsed_time_per_iteration},
                                 iteration)
        log_string = ' iteration {:8d}/{:8d} |'.format(
            iteration, args.train_iters)
        log_string += ' consumed samples: {:12d} |'.format(
            args.consumed_train_samples)
        log_string += ' consumed tokens: {:12d} |'.format(
            args.consumed_train_tokens)
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
        log_string += ' learning rate: {:.3E} |'.format(learning_rate)
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
                total_loss_dict[key] = torch.cuda.FloatTensor([0.0])
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
        if iteration == args.log_interval:
            elapsed_time_per_iteration_10 = 0.0
            tokens_per_second_10 = 0.0
            tflops_10 = 0.0
            times = 0
            tps_per_device = 0.0
        if iteration >= 4:
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
        if report_memory_flag and learning_rate > 0.:
            # Report memory after optimizer state has been initialized.
            if torch.distributed.get_rank() == 0:
                num_microbatches = get_num_microbatches()
                report_theoretical_memory(args, num_microbatches=num_microbatches, verbose=True)
            report_memory('(after {} iterations)'.format(iteration))
            report_memory_flag = False
        timers.log(timers_to_log, normalizer=args.log_interval)

    return report_memory_flag


def save_checkpoint_and_time(iteration, model, optimizer, opt_param_scheduler):
    timers = get_timers()
    # Extra barrier is added to make sure
    # all ranks report the max time.
    timers('save-checkpoint', log_level=0).start(barrier=True)
    save_checkpoint(iteration, model, optimizer, opt_param_scheduler)
    timers('save-checkpoint').stop(barrier=True)
    timers.log(['save-checkpoint'])


def train(forward_step_func, model, optimizer, opt_param_scheduler,
          train_data_iterator, valid_data_iterator,
          process_non_loss_data_func, config):
    """Train the model function."""
    args = get_args()
    timers = get_timers()

    # Write args to tensorboard
    write_args_to_tensorboard()

    # Turn on training mode which enables dropout.
    for model_module in model:
        model_module.train()

    # Tracking loss.
    total_loss_dict = {}

    # Iterations.
    iteration = args.iteration

    # Translate args to core configuration
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
        if args.delay_grad_reduce:
            config.grad_sync_func = [model_chunk.start_grad_sync for model_chunk in model]
            if len(model) == 1:
                config.grad_sync_func = config.grad_sync_func[0]
    if args.overlap_param_gather and args.delay_param_gather:
        config.param_sync_func = [lambda x: optimizer.finish_param_sync(model_index, x)
                                  for model_index in range(len(model))]
        if len(model) == 1:
            config.param_sync_func = config.param_sync_func[0]
    config.finalize_model_grads_func = finalize_model_grads

    timers('interval-time', log_level=0).start(barrier=True)
    print_datetime('before the start of training step')
    report_memory_flag = True
    exit = False

    if args.manual_gc:
        # Disable the default garbage collector and perform the collection manually.
        # This is to align the timing of garbage collection across ranks.
        assert args.manual_gc_interval >= 0, \
            'Manual garbage collection interval should be laerger than or equal to 0.'
        gc.disable()
        gc.collect()

    while iteration < args.train_iters:
        if args.profile and \
           iteration == args.profile_step_start and \
           torch.distributed.get_rank() in args.profile_ranks:
            torch.cuda.cudart().cudaProfilerStart()
            torch.autograd.profiler.emit_nvtx(record_shapes=True).__enter__()

        update_num_microbatches(args.consumed_train_samples)
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
        args.curr_iteration = iteration
        loss_dict, skipped_iter, grad_norm, num_zeros_in_grad = \
            train_step(forward_step_func,
                       train_data_iterator,
                       model,
                       optimizer,
                       opt_param_scheduler,
                       config)
        iteration += 1
        args.iteration = iteration
        new_samples = mpu.get_data_parallel_world_size() * \
                                       args.micro_batch_size * \
                                       get_num_microbatches()
        args.consumed_train_samples += new_samples
        # This actual_seq_length is used for actual consumed tokens calculation, flops calculation, and logging.
        args.actual_seq_length = args.seq_length
        if args.curriculum_learning_legacy or args.data_efficiency_curriculum_learning:
            args.actual_seq_length = args.curriculum_seqlen
        if args.random_ltd:
            args.random_ltd_reserved_length = model[0].random_ltd_scheduler.get_current_seq()
            if args.random_ltd_reserved_length < args.actual_seq_length:
                args.actual_seq_length = (args.actual_seq_length * (args.num_layers - args.random_ltd_layer_num) + args.random_ltd_reserved_length * args.random_ltd_layer_num) // args.num_layers
        if args.curriculum_learning_legacy or args.data_efficiency_curriculum_learning:
            if hasattr(args, 'data_efficiency_curriculum_learning_numel'):
                act_mbsz = args.data_efficiency_curriculum_learning_numel / args.curriculum_seqlen
                act_token = act_mbsz * args.actual_seq_length
                args.consumed_train_tokens += mpu.get_data_parallel_world_size() * \
                        get_num_microbatches() * act_token
            else:
                args.consumed_train_tokens += new_samples * args.actual_seq_length
        else:
            args.consumed_train_tokens += new_samples * args.actual_seq_length
        
        # Logging.
        if args.deepspeed:
            if hasattr(model[0].optimizer, 'cur_scale'):
                loss_scale = model[0].optimizer.cur_scale
            else:
                loss_scale = None
        else:
            loss_scale = optimizer.get_loss_scale().item()
        params_norm = None
        if args.log_params_norm:
            params_norm = calc_params_l2_norm(model)
        report_memory_flag = training_log(loss_dict, total_loss_dict,
                                          optimizer.param_groups[0]['lr'],
                                          iteration, loss_scale,
                                          report_memory_flag, skipped_iter,
                                          grad_norm, params_norm, num_zeros_in_grad,
                                          model, optimizer)

        # Autoresume
        if args.adlr_autoresume and \
           (iteration % args.adlr_autoresume_interval == 0):
            check_adlr_autoresume_termination(iteration, model, optimizer,
                                              opt_param_scheduler)

        # Evaluation
        if args.eval_interval and iteration % args.eval_interval == 0 and \
           args.do_valid:
            timers('interval-time').stop()
            if args.manual_gc and args.manual_gc_eval:
                # Collect all objects.
                gc.collect()
            prefix = 'iteration {}'.format(iteration)
            evaluate_and_print_results(prefix, forward_step_func,
                                       valid_data_iterator, model,
                                       iteration, process_non_loss_data_func,
                                       config, False)
            if args.manual_gc and args.manual_gc_eval:
                # Collect only the objects created and used in evaluation.
                gc.collect(generation=0)
            timers('interval-time', log_level=0).start(barrier=True)

        # Checkpointing
        saved_checkpoint = False
        if args.exit_signal_handler:
            signal_handler = get_signal_handler()
            if any(signal_handler.signals_received()):
                save_checkpoint_and_time(iteration, model, optimizer,
                                         opt_param_scheduler)
                print_datetime('exiting program after receiving SIGTERM.')
                exit = True
                break

        if args.save and args.save_interval and \
           iteration % args.save_interval == 0:
            timers('interval-time').stop()
            save_checkpoint_and_time(iteration, model, optimizer,
                                     opt_param_scheduler)
            saved_checkpoint = True
            timers('interval-time', log_level=0).start(barrier=True)

        # Exiting based on duration
        if args.exit_duration_in_mins:
            train_time = (time.time() - _TRAIN_START_TIME) / 60.0
            done_cuda = torch.cuda.IntTensor(
                [train_time > args.exit_duration_in_mins])
            torch.distributed.all_reduce(
                done_cuda, op=torch.distributed.ReduceOp.MAX)
            done = done_cuda.item()
            if done:
                if not saved_checkpoint:
                    save_checkpoint_and_time(iteration, model, optimizer,
                                             opt_param_scheduler)
                print_datetime('exiting program after {} minutes'.format(train_time))
                exit = True
                break

        # Exiting based on iterations
        if args.exit_interval and iteration % args.exit_interval == 0:
            if args.save and not saved_checkpoint:
                save_checkpoint_and_time(iteration, model, optimizer,
                                         opt_param_scheduler)
            torch.distributed.barrier()
            print_datetime('exiting program at iteration {}'.format(iteration))
            exit = True
            break

        if args.profile and \
           iteration == args.profile_step_end and \
           torch.distributed.get_rank() in args.profile_ranks:
            torch.cuda.cudart().cudaProfilerStop()

        if args.manual_gc:
            if args.manual_gc_interval != 0 and iteration % args.manual_gc_interval == 0:
                gc.collect()

    # Flush TensorBoard and WandB writers.
    writer = get_tensorboard_writer()
    if writer:
        writer.flush()
    wandb_writer = get_wandb_writer()
    if wandb_writer:
        wandb_writer.finish()

    # If any exit conditions (signal handler, duration, iterations) have been reached, exit.
    if exit:
        sys.exit()

    return iteration


def evaluate(forward_step_func,
             data_iterator,
             model,
             process_non_loss_data_func,
             config,
             verbose=False):
    """Evaluation."""
    args = get_args()
    timers = get_timers()

    timers('evaluate', log_level=0).start(barrier=True)

    if args.vision_pretraining and args.vision_pretraining_type == "dino":
        compute_feature_bank(model)

    # Turn on evaluation mode which disables dropout.
    for model_module in model:
        model_module.eval()

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
                loss_dicts = forward_backward_func(
                    forward_step_func=forward_step_func,
                    data_iterator=data_iterator,
                    model=model,
                    num_microbatches=get_num_microbatches(),
                    seq_length=args.seq_length,
                    micro_batch_size=args.micro_batch_size,
                    decoder_seq_length=args.decoder_seq_length,
                    forward_only=True)
            config.timers = get_timers()

            # Empty unused memory
            if args.empty_unused_memory_level >= 1:
                torch.cuda.empty_cache()

            if mpu.is_pipeline_last_stage(ignore_virtual=True):
                # Reduce across processes.
                for loss_dict in loss_dicts:
                    for key in loss_dict:
                        total_loss_dict[key] = total_loss_dict.get(
                            key, torch.cuda.FloatTensor([0.0])) + loss_dict[key]

            args.consumed_valid_samples += eval_batch_size

            if args.exit_duration_in_mins:
                train_time = (time.time() - _TRAIN_START_TIME) / 60.0
                done_cuda = torch.cuda.IntTensor(
                    [train_time > args.exit_duration_in_mins])
                torch.distributed.all_reduce(
                    done_cuda, op=torch.distributed.ReduceOp.MAX)
                done = done_cuda.item()
                if done:
                    print_rank_0('Exiting during evaluation, timelimit reached')
                    return None, None, True

        collected_non_loss_data = None
        if process_non_loss_data_func is not None and is_last_rank():
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
        total_loss_dict[key] /= args.eval_iters * eval_num_microbatches

    timers('evaluate').stop()
    timers.log(['evaluate'])

    return total_loss_dict, collected_non_loss_data, False

def evaluate_and_print_results(prefix, forward_step_func,
                               data_iterator, model,
                               iteration, process_non_loss_data_func, config,
                               verbose=False, write_to_tensorboard=True, test=False):
    """Helper function to evaluate and dump results on screen."""
    args = get_args()
    if write_to_tensorboard:
        writer = get_tensorboard_writer()
    else:
        writer = None

    wandb_writer = get_wandb_writer()

    total_loss_dict, collected_non_loss_data, timelimit = evaluate(
        forward_step_func, data_iterator, model,
        process_non_loss_data_func, config, verbose)
    # Timelimit hit during evaluation
    if timelimit:
        return
    string = ' validation loss at {} | '.format(prefix)
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


def cyclic_iter(iter):
    while True:
        for x in iter:
            yield x


def build_train_valid_test_datasets(build_train_valid_test_datasets_provider):
    """Build pretraining datasets."""

    args = get_args()

    # Number of train/valid/test samples.
    if args.train_samples:
        train_samples = args.train_samples
    else:
        train_samples = args.train_iters * args.global_batch_size
    eval_iters = (args.train_iters // args.eval_interval + 1) * \
                 args.eval_iters
    test_iters = args.eval_iters
    train_val_test_num_samples = [train_samples,
                                  eval_iters * args.global_batch_size,
                                  test_iters * args.global_batch_size]
    print_rank_0(' > datasets target sizes (minimum size):')
    print_rank_0('    train:      {}'.format(train_val_test_num_samples[0]))
    print_rank_0('    validation: {}'.format(train_val_test_num_samples[1]))
    print_rank_0('    test:       {}'.format(train_val_test_num_samples[2]))

    # Build the datasets.
    return build_train_valid_test_datasets_provider(train_val_test_num_samples)


def build_train_valid_test_data_loaders(
        build_train_valid_test_datasets_provider):
    """Build pretraining data loaders."""

    args = get_args()

    (train_dataloader, valid_dataloader, test_dataloader) = (None, None, None)

    print_rank_0('> building train, validation, and test datasets ...')

    # Backward compatibility, assume fixed batch size.
    if args.iteration > 0 and args.consumed_train_samples == 0:
        assert args.train_samples is None, \
            'only backward compatiblity support for iteration-based training'
        args.consumed_train_samples = args.iteration * args.global_batch_size
    if args.iteration > 0 and args.consumed_valid_samples == 0:
        if args.train_samples is None:
            args.consumed_valid_samples = (args.iteration // args.eval_interval) * \
                args.eval_iters * args.global_batch_size

    # Rely on distributed-aware core datasets, temporary
    is_distributed = getattr(build_train_valid_test_datasets_provider, "is_distributed", False)

    # Construct the data pipeline
    if is_distributed or mpu.get_tensor_model_parallel_rank() == 0:

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
        flags = torch.cuda.LongTensor(
            [int(do_train), int(do_valid), int(do_test)])
    else:
        flags = torch.cuda.LongTensor([0, 0, 0])

    torch.distributed.broadcast(flags, 0)

    args.do_train = getattr(args, "do_train", False) or flags[0].item()
    args.do_valid = getattr(args, "do_valid", False) or flags[1].item()
    args.do_test = getattr(args, "do_test", False) or flags[2].item()

    return train_dataloader, valid_dataloader, test_dataloader


def build_train_valid_test_data_iterators(
        build_train_valid_test_datasets_provider):
    """Build pretraining data iterators."""

    args = get_args()

    # Build loaders.
    train_dataloader, valid_dataloader, test_dataloader = \
        build_train_valid_test_data_loaders(
            build_train_valid_test_datasets_provider)

    # Build iterators.
    dl_type = args.dataloader_type
    assert dl_type in ['single', 'cyclic']

    if train_dataloader is not None:
        train_data_iterator = iter(train_dataloader) if dl_type == 'single' \
                              else iter(cyclic_iter(train_dataloader))
    else:
        train_data_iterator = None

    if valid_dataloader is not None:
        valid_data_iterator = iter(valid_dataloader) if dl_type == 'single' \
                              else iter(cyclic_iter(valid_dataloader))
    else:
        valid_data_iterator = None

    if test_dataloader is not None:
        test_data_iterator = iter(test_dataloader) if dl_type == 'single' \
                             else iter(cyclic_iter(test_dataloader))
    else:
        test_data_iterator = None

    return train_data_iterator, valid_data_iterator, test_data_iterator
