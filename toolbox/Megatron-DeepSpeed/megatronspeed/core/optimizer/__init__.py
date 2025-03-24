import logging
from functools import wraps
from typing import Callable, Dict, List, Optional

import torch

try:
    from apex.optimizers import FusedAdam as Adam
    from apex.optimizers import FusedSGD as SGD
except ImportError:
    import warnings

    warnings.warn(
        f'Transformer Engine and Apex are not installed. Falling back to Torch optimizers.'
    )

    ## apex's FusedAdam is a drop-in replacement for torch's AdamW
    ## see https://github.com/NVIDIA/apex/blob/7b73b12361068a10b0f44844534613f252a5ea75/apex/optimizers/fused_adam.py#L16
    from torch.optim import AdamW as Adam, SGD

from megatron.core import mpu

from megatron.training.global_vars import get_args
from megatron.core.distributed import ParamAndGradBuffer
from megatron.core.transformer.module import MegatronModule
from megatron.core.utils import log_single_rank
from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
from megatron.core.optimizer.grad_scaler import ConstantGradScaler, DynamicGradScaler
from megatron.core.optimizer.optimizer import (
    ChainedOptimizer,
    Float16OptimizerWithFloat16Params,
    FP32Optimizer,
    MegatronOptimizer,
)
from megatron.core.optimizer.optimizer_config import OptimizerConfig

from megatron.core.optimizer import (
    logger,
    _get_param_groups,
    _update_min_and_max_lr_in_param_groups
)


def get_param_groups(modules,
                     no_weight_decay_cond,
                     scale_lr_cond,
                     lr_mult):
    """creates param groups based on weight decay condition (regularized vs non regularized)
       and learning rate scale condition (args.lr vs lr_mult * args.lr)
       scale_lr_cond is used during finetuning where head of the network requires a scaled
       version of the base learning rate. 
    """
    wd_no_scale_lr = []
    wd_scale_lr = []
    no_wd_no_scale_lr = []
    no_wd_scale_lr = []
    for module in modules:
        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue

            if no_weight_decay_cond is not None:
                no_wd = no_weight_decay_cond(name, param)
            else:
                # do not regularize biases nor Norm parameters
                no_wd = name.endswith(".bias") or len(param.shape) == 1

            if scale_lr_cond is not None:
                scale_lr = scale_lr_cond(name, param)
            else:
                scale_lr = False

            if not no_wd and not scale_lr:
                wd_no_scale_lr.append(param)
            elif not no_wd and scale_lr:
                wd_scale_lr.append(param)
            elif no_wd and not scale_lr:
                no_wd_no_scale_lr.append(param)
            else:
                no_wd_scale_lr.append(param)

    param_groups = []
    if len(wd_no_scale_lr):
        param_groups.append({'name': 'wd_no_scale_lr', 'params': wd_no_scale_lr, 'wd_mult': 1.0, 'lr_mult': 1.0})
    if len(wd_scale_lr):
        param_groups.append({'name': 'wd_scale_lr', 'params': wd_scale_lr, 'wd_mult': 1.0, 'lr_mult': lr_mult})
    if len(no_wd_no_scale_lr):
        param_groups.append({'name': 'no_wd_no_scale_lr', 'params': no_wd_no_scale_lr, 'wd_mult': 0.0, 'lr_mult': 1.0})
    if len(no_wd_scale_lr):
        param_groups.append({'name': 'no_wd_scale_lr', 'params': no_wd_scale_lr, 'wd_mult': 0.0, 'lr_mult': lr_mult})

    return param_groups

def _get_param_groups_mod(
    model_chunks: List[MegatronModule],
    no_weight_decay_cond: Callable,
    scale_lr_cond: Callable,
    lr_mult: float,
    use_decoupled_learning_rate: bool,
) -> List[Dict]:
    """Create parameter groups for optimizer.

    Creates parameter groups based on weight decay condition (regularized vs
    non regularized), learning rate scale condition (lr vs lr_mult * lr),
    and whether it is expert parameters. scale_lr_cond is used during finetuning
    where head of the network requires a scaled version of the base learning rate.

    Args:
        model_chunks (List[MegatronModule]): model chunks to create parameter
            groups for.
        no_weight_decay_cond (func): function to determine whether a parameter
            should not perform weight decay.
        scale_lr_cond (func): function to determine whether a parameter
            should have a scaled learning rate.
        lr_mult (float): learning rate multiplier for parameters that
            satisfy scale_lr_cond.
        use_decoupled_learning_rate (bool): true if using decoupled learning rate.

    Returns:
        List of parameter groups.
    """

    # Map (wd_mult, lr_mult, is_expert_parallel, is_decoupled_lr) to params.
    params_map = {}
    for model_chunk in model_chunks:
        for name, param in model_chunk.named_parameters():
            if not param.requires_grad:
                continue

            is_expert_parallel = not getattr(param, 'allreduce', True)

            if no_weight_decay_cond is not None:
                no_wd = no_weight_decay_cond(name, param)
            else:
                # Do not regularize biases and norm parameters.
                no_wd = name.endswith(".bias") or len(param.shape) == 1

            if scale_lr_cond is not None:
                scale_lr = scale_lr_cond(name, param)
            else:
                scale_lr = False

            if not no_wd and not scale_lr:
                wd_mult, _lr_mult = 1.0, 1.0
            elif not no_wd and scale_lr:
                wd_mult, _lr_mult = 1.0, lr_mult
            elif no_wd and not scale_lr:
                wd_mult, _lr_mult = 0.0, 1.0
            else:
                wd_mult, _lr_mult = 0.0, lr_mult

            is_decoupled_lr = False
            # For input/embedding and output layer: embedding.word_embeddings.weight / output_layer.weight.
            if use_decoupled_learning_rate and getattr(
                param, 'is_embedding_or_output_parameter', False
            ):
                is_decoupled_lr = True

            key = (wd_mult, _lr_mult, is_expert_parallel, is_decoupled_lr)
            if key not in params_map:
                params_map[key] = []
            params_map[key].append(param)

    param_groups = []
    for (wd_mult, _lr_mult, is_expert_parallel, is_decoupled_lr), params in params_map.items():
        assert len(params) > 0
        if wd_mult == 1.0 and _lr_mult == 1.0:
            name = 'wd_no_scale_lr'
        elif wd_mult == 1.0 and _lr_mult == lr_mult:
            name = 'wd_scale_lr'
        elif wd_mult == 0.0 and _lr_mult == 1.0:
            name = 'no_wd_no_scale_lr'
        else:
            name = 'no_wd_scale_lr'
        param_groups.append(
            {
                'name': name,
                'params': params,
                'wd_mult': wd_mult,
                'lr_mult': _lr_mult,
                'is_expert_parallel': is_expert_parallel,
                'is_decoupled_lr': is_decoupled_lr,
            }
        )

    return param_groups

def get_megatron_optimizer_wrapper(get_megatron_optimizer):
    @wraps(get_megatron_optimizer)
    def wrapper(
        config: OptimizerConfig,
        model_chunks: List[MegatronModule],
        no_weight_decay_cond: Optional[Callable] = None,
        scale_lr_cond: Optional[Callable] = None,
        lr_mult: float = 1.0,
        ):
        args = get_args()
        
        if not args.deepspeed:
            return get_megatron_optimizer(
                    config,
                    model_chunks,
                    no_weight_decay_cond,
                    scale_lr_cond,
                    lr_mult)

        log_single_rank(logger, logging.INFO, f'Setting up optimizer with config {config}')

        args = get_args()

        # Base optimizer.
        param_groups = _get_param_groups(
            model_chunks,
            no_weight_decay_cond,
            scale_lr_cond,
            lr_mult,
            use_decoupled_learning_rate=config.decoupled_lr is not None,
        )
        param_groups = _update_min_and_max_lr_in_param_groups(
            param_groups,
            lr=config.lr,
            min_lr=config.min_lr,
            decoupled_lr=config.decoupled_lr,
            decoupled_min_lr=config.decoupled_min_lr,
        )
        if args.create_moe_param_group:
            from deepspeed.moe.utils import split_params_into_different_moe_groups_for_optimizer
            param_groups = split_params_into_different_moe_groups_for_optimizer(param_groups)

        if args.cpu_optimizer:
            assert args.optimizer == 'adam', 'CPU offloading is for Adam'
            if args.cpu_torch_adam:
                cpu_adam_optimizer = torch.optim.AdamW
            else:
                from deepspeed.ops.adam import DeepSpeedCPUAdam
                cpu_adam_optimizer = DeepSpeedCPUAdam
            optimizer = cpu_adam_optimizer(param_groups,
                                        lr=args.lr,
                                        weight_decay=args.weight_decay,
                                        betas=(args.adam_beta1, args.adam_beta2),
                                        eps=args.adam_eps)
        else:
            if args.optimizer == 'adam':
                if args.ds_fused_adam:
                    global Adam
                    from deepspeed.ops.adam import FusedAdam
                    Adam = FusedAdam
                optimizer = Adam(param_groups,
                                lr=args.lr,
                                weight_decay=args.weight_decay,
                                betas=(args.adam_beta1, args.adam_beta2),
                                eps=args.adam_eps)
            elif args.optimizer == 'sgd':
                optimizer = SGD(param_groups,
                                lr=args.lr,
                                weight_decay=args.weight_decay,
                                momentum=args.sgd_momentum)
            else:
                raise Exception('{} optimizer is not supported.'.format(
                args.optimizer))

        if args.deepspeed:
            return optimizer
    return wrapper
