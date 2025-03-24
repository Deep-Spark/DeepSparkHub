"""General utilities."""

import sys
import os

import torch
from torch.nn.parallel import DistributedDataParallel as torchDDP

from deepspeed.accelerator import get_accelerator

from megatron.core import mpu
from megatron.training import (
    get_args,
    get_adlr_autoresume,
)
from megatron.core.num_microbatches_calculator import get_num_microbatches
from megatron.training.utils import print_rank_0
from megatronspeed.legacy.model.rotary_pos_embedding import RotaryEmbedding

def update_rotary_pos_emb(seq_length):
    args = get_args()
    rotary_dim = args.hidden_size // args.num_attention_heads \
        if args.kv_channels is None else args.kv_channels

    if args.rotary_percent < 1.0:
        rotary_dim = int(rotary_dim * args.rotary_percent)

    # partial rotary embeddings, which is better than full rotary
    # Wang and Komatsuzaki et al
    # https://github.com/kingoflolz/mesh-transformer-jax/
    rotary_pos_emb = RotaryEmbedding(rotary_dim, theta=args.rope_theta)(seq_length).to(
        get_accelerator().current_device_name())
    args.rotary_pos_emb = rotary_pos_emb


def get_ltor_masks_and_position_ids(data,
                                    eod_token,
                                    reset_position_ids,
                                    reset_attention_mask,
                                    eod_mask_loss,
                                    skip_mask=False):
    """Build masks and position id for left to right model."""

    # Extract batch size and sequence length.
    micro_batch_size, seq_length = data.size()

    # Attention mask (lower triangular).
    if reset_attention_mask:
        att_mask_batch = micro_batch_size
    else:
        att_mask_batch = 1
    attention_mask = None
    if not skip_mask:
        attention_mask = torch.tril(torch.ones(
            (att_mask_batch, seq_length, seq_length), device=data.device)).view(
                att_mask_batch, 1, seq_length, seq_length)

    # Loss mask.
    loss_mask = torch.ones(data.size(), dtype=torch.float, device=data.device)
    if eod_mask_loss:
        loss_mask[data == eod_token] = 0.0

    # Position ids.
    position_ids = torch.arange(seq_length, dtype=torch.long,
                                device=data.device)
    position_ids = position_ids.unsqueeze(0).expand_as(data)
    # We need to clone as the ids will be modifed based on batch index.
    if reset_position_ids:
        position_ids = position_ids.clone()

    if reset_position_ids or reset_attention_mask:
        # Loop through the batches:
        for b in range(micro_batch_size):

            # Find indecies where EOD token is.
            eod_index = position_ids[b, data[b] == eod_token]
            # Detach indecies from positions if going to modify positions.
            if reset_position_ids:
                eod_index = eod_index.clone()

            # Loop through EOD indecies:
            prev_index = 0
            for j in range(eod_index.size()[0]):
                i = eod_index[j]
                # Mask attention loss.
                if reset_attention_mask and not skip_mask:
                    attention_mask[b, 0, (i + 1):, :(i + 1)] = 0
                # Reset positions.
                if reset_position_ids:
                    position_ids[b, (i + 1):] -= (i + 1 - prev_index)
                    prev_index = i + 1

    # Convert attention mask to binary:
    if not skip_mask:
        attention_mask = (attention_mask < 0.5)

    return attention_mask, loss_mask, position_ids


def is_aml():
    # Are we running inside an Azure Machine Learning (AML) environment?
    return 'AZUREML_EXPERIMENT_ID' in os.environ


def is_rank_0():
    """Check whether it is rank 0. For AML, check if it is rank 0 of a node"""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0 or (
            is_aml() and torch.distributed.get_rank() % get_accelerator().device_count() == 0
            ):
            return True
        else:
            return False
    else:
        return True

def get_parameters_in_billions(model):
    gpus_per_model = torch.distributed.get_world_size(group=mpu.get_model_parallel_group())

    approx_parameters_in_billions = sum([sum([p.ds_numel if hasattr(p,'ds_id') else  p.nelement() for p in model_module.parameters()])
                                        for model_module in model])

    return approx_parameters_in_billions*gpus_per_model/(1e9)

def throughput_calculator(model, args, iteration_time, total_iterations):
    batch_size = args.micro_batch_size * get_num_microbatches() * args.data_parallel_size
    approx_parameters_in_billions = None if (model is None) else get_parameters_in_billions(model)
    elapsed_time_per_iter = iteration_time/total_iterations
    samples_per_second = batch_size / elapsed_time_per_iter

    #flops calculator
    hidden_size = args.hidden_size
    num_layers = args.num_layers
    vocab_size = args.padded_vocab_size

    # General TFLOPs formula (borrowed from Equation 3 in Section 5.1 of
    # https://arxiv.org/pdf/2104.04473.pdf).
    # The factor of 4 is when used with activation check-pointing,
    # otherwise it will be 3.
    checkpoint_activations_factor = 3
    if hasattr(args, 'checkpoint_activations') and args.checkpoint_activations:
        checkpoint_activations_factor = 4
    if hasattr(args, 'recompute_granularity') and args.recompute_granularity == 'selective':
        checkpoint_activations_factor = 4
    seq_len = args.seq_length
    if hasattr(args, 'actual_seq_length'):
        seq_len = args.actual_seq_length
    flops_per_iteration = (24 * checkpoint_activations_factor * batch_size * seq_len * num_layers * (hidden_size**2)) * (1. + (seq_len / (6. * hidden_size)) + (vocab_size / (16. * num_layers * hidden_size)))
    

    def execCmd(cmd):
        r = os.popen(cmd)
        text = r.read()
        r.close()
        return text

    # IS_BI_V150 = "BI-V150" in execCmd("ixsmi -L")
    IS_BI_V150 = True
    if IS_BI_V150:
        tflops = flops_per_iteration / (elapsed_time_per_iter * (args.world_size / 2) * (10**12))
    else:
        tflops = flops_per_iteration / (elapsed_time_per_iter * args.world_size * (10**12))
    return samples_per_second, tflops, approx_parameters_in_billions

def checkpoint_throughput_calculator(model, latency_second):
    approx_parameters_in_billions = get_parameters_in_billions(model)
    checkpoint_multiplier = 14  # fp16 weights (2), fp32 weights (4), fp32 momentum (4), fp32 variance (4)
    checkpoint_GB = approx_parameters_in_billions * checkpoint_multiplier
    GB_per_second = checkpoint_GB / latency_second
    print_rank_0(f"Checkpoint Save GB: {round(checkpoint_GB, 3)}, GB/Sec: {round(GB_per_second,2)}, Latency(second): {round(latency_second, 3)}")


def get_fingerprint_header():
    return f"{'min':^13} {'max':^13} {'mean':^13} {'l2 norm':^12} metadata"

def get_fingerprint(p):
    return f"{p.min():13.6e} {p.max():13.6e} {p.mean():13.6e} {p.norm():12.6e}"


def dump_position_embed_weights(preamble, iteration, model):
    # return 
    from deepspeed.utils import safe_get_full_fp32_param
    tp_rank = mpu.get_tensor_model_parallel_rank()
    pp_rank = mpu.get_pipeline_model_parallel_rank()
    dp_rank = mpu.get_data_parallel_rank()
    get_fingerprint_header()
    for n, p in model[0].named_parameters():
        if 'position_embeddings' in n:
            tag = "pos_embed"
        elif "word_embeddings" in n:
            tag = "word_embed"
        else:
            continue 
        print(f"iter {iteration} {preamble} {tag} lp {tp_rank}/{pp_rank}/{dp_rank}: {get_fingerprint(p)} {p.shape}\n")
        fp32_value = safe_get_full_fp32_param(p)
        if fp32_value is not None: 
            print(f"iter {iteration} {preamble} {tag} hp {tp_rank}/{pp_rank}/{dp_rank}: {get_fingerprint(fp32_value)} {p.shape}\n")

def dump_weights(preamble, iteration, model, optimizer, tensor=None):
    # return
    tp_rank = mpu.get_tensor_model_parallel_rank()
    pp_rank = mpu.get_pipeline_model_parallel_rank()
    dp_rank = mpu.get_data_parallel_rank()
    dp_size = mpu.get_data_parallel_world_size()
    fn = f"debug-bf16-{iteration}-pp{pp_rank}-tp{tp_rank}-dp{dp_rank}-{preamble}.txt"

    # only care for first and last pp stages and dp0 tp0
    #if not (mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage()):
    #    return

    #if not (tp_rank == 0 and dp_rank == 0):
    #    return

    if tensor is not None:
        orig_tensor = tensor
        if hasattr(tensor, "_hp_param"):
            numel = tensor._hp_param.numel() # // dp_size
            tensor = tensor.flatten().narrow(0, 0, numel)

    #print(fn)
    with open(fn, "w") as fh:
        fh.write(f"{get_fingerprint_header()}\n")

        if tensor is not None:
            fh.write(f"{get_fingerprint(tensor)} tensor {tensor.shape}\n")
        else:
            for n, p in model[0].named_parameters():
                fh.write(f"{get_fingerprint(p)} {n} {p.shape}\n")


    return


    # until we figure out how to dump the actual fp32 values don't do this
    fn = f"debug-fp32-{iteration}-pp{pp_rank}-tp{tp_rank}-dp{dp_rank}-{preamble}.txt"
    with open(fn, "w") as fh:
        fh.write(f"{get_fingerprint_header()}\n")
        if tensor is not None:
            tensor = orig_tensor
            if hasattr(tensor, "_hp_param"):
                fh.write(f"{get_fingerprint(tensor._hp_param)} tensor {tensor._hp_param.shape}\n")
                #fh.write(f"{get_fingerprint(tensor._hp_grad)} tensor grad\n")
            else:
                fh.write(f"{get_fingerprint(tensor)} tensor {tensor.shape}\n")
                #fh.write(f"{get_fingerprint(tensor.grad)} tensor grad\n")

        else:
            if hasattr(model[0].module.tied_modules, "embed"):
                p = model[0].module.tied_modules.embed.word_embeddings.weight._hp_param
                fh.write(f"{get_fingerprint(p)} module.tied_modules.embed.word_embeddings.weight._hp_param {p.shape}\n")


