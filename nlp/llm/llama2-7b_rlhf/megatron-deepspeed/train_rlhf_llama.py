import os
import torch
from torch import Tensor
from functools import partial
from typing import Union
from megatron_ds import get_args, get_rlhf_args
from megatron_ds import print_rank_0
from megatron_ds import get_timers
from megatron_ds import get_tokenizer
from megatron_ds.core import mpu, tensor_parallel
from megatron_ds.core.enums import ModelType
from megatron_ds.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron_ds.core.datasets.blended_megatron_dataset_config import GPTDatasetConfig
from megatron_ds.core.datasets.gpt_dataset import GPTDataset
import megatron_ds.model
from megatron_ds.model import GPTModel
from megatron_ds.utils import (
    get_ltor_masks_and_position_ids,
    get_batch_on_this_cp_rank,
    average_losses_across_data_parallel_group
)
from megatron_ds.arguments import core_transformer_config_from_args

from megatron_ds.rlhf.training_rlhf import RLHFPPOTrainer



def model_provider(pre_process=True, post_process=True, rlhf_training=False) -> Union[GPTModel, megatron_ds.model.GPTModel]:
    """Builds the model.

    If you set the use_mcore_models to True, it will return the mcore GPT model and if not the legacy GPT model.

    Args:
        pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
        post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.


    Returns:
        Union[GPTModel, megatron_ds.model.GPTModel]: The returned model
    """
    if rlhf_training:
        args = get_rlhf_args()
    else:
        args = get_args()

    print_rank_0('building GPT model ...')
    config = core_transformer_config_from_args(args)

    assert(args.context_parallel_size == 1), "Context parallelism is only supported with Megatron Core!"

    model = megatron_ds.model.GPTModel(
        config,
        num_tokentypes=0,
        parallel_output=True,
        pre_process=pre_process,
        post_process=post_process,
        rlhf_training=rlhf_training
    )

    return model


def get_batch(data_iterator):
    """Generate a batch."""

    # TODO: this is pretty hacky, find a better way
    if (not mpu.is_pipeline_first_stage()) and (not mpu.is_pipeline_last_stage()):
        return None, None, None, None, None

    args = get_args()
    tokenizer = get_tokenizer()

    # Items and their type.
    keys = ['text']
    datatype = torch.int64

    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    data_b = tensor_parallel.broadcast_data(keys, data, datatype)

    # Unpack.
    tokens_ = data_b['text'].long()
    labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()

    # Get the masks and postition ids.
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        tokenizer.eod,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss)

    batch = {
        'tokens': tokens,
        'labels': labels,
        'loss_mask': loss_mask,
        'attention_mask': attention_mask,
        'position_ids': position_ids
    }
    # slice batch along sequence dimension for context parallelism
    batch = get_batch_on_this_cp_rank(batch)

    return batch.values()


def loss_func(loss_mask: Tensor, output_tensor: Tensor):
    """Loss function.

    Args:
        loss_mask (Tensor): Used to mask out some portions of the loss
        output_tensor (Tensor): The tensor with the losses
    """    
    args = get_args()

    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    if args.context_parallel_size > 1:
        loss = torch.cat([torch.sum(losses.view(-1) * loss_mask).view(1), loss_mask.sum().view(1)])
        torch.distributed.all_reduce(loss, group=mpu.get_context_parallel_group())
        loss = loss[0] / loss[1]
    else:
        loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

    # Check individual rank losses are not NaN prior to DP all-reduce.
    if args.check_for_nan_in_loss_and_grad:
        global_rank = torch.distributed.get_rank()
        assert not loss.isnan(), (
            f'Rank {global_rank}: found NaN in local forward loss calculation. '
            f'Device: {torch.cuda.current_device()}, node: {os.uname()[1]}'
        )

    # Reduce loss for logging.
    averaged_loss = average_losses_across_data_parallel_group([loss])

    return loss * args.context_parallel_size, {'lm loss': averaged_loss[0]}


def is_dataset_built_on_rank():
    return (mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage()) and mpu.get_tensor_model_parallel_rank() == 0


def core_gpt_dataset_config_from_args(args):
    return GPTDatasetConfig(
        is_built_on_rank=is_dataset_built_on_rank,
        random_seed=args.seed,
        sequence_length=args.max_prompt_seq_len,
        blend=args.data_path,
        blend_per_split=[args.train_data_path, args.valid_data_path, args.test_data_path],
        split=args.split,
        path_to_cache=args.data_cache_path,
        return_document_ids=args.retro_return_doc_ids
    )


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build the train test and validation datasets.

    Args:
        train_val_test_num_samples : A list containing the number of samples in train test and validation.
    """
    args = get_args()

    print_rank_0("> building train, validation, and test datasets for GPT ...")

    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        GPTDataset,
        train_val_test_num_samples,
        core_gpt_dataset_config_from_args(args)
    ).build()

    print_rank_0("> finished creating GPT datasets ...")

    return train_ds, valid_ds, test_ds


if __name__ == "__main__":

    # Temporary for transition to core datasets
    train_valid_test_datasets_provider.is_distributed = True

    trainer = RLHFPPOTrainer(train_valid_test_datasets_provider,
                            model_provider,
                            ModelType.encoder_or_decoder)
