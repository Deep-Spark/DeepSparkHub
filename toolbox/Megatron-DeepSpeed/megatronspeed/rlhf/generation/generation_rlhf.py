# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Generation utilities."""

import torch
import torch.nn.functional as F

from megatron.training import get_tokenizer
from megatron.core import mpu, tensor_parallel
from megatron.training.utils import get_ltor_masks_and_position_ids
from .communication_rlhf import (
    copy_from_last_to_first_pipeline_stage,
    broadcast_float_list, broadcast_int_list,
    broadcast_tensor)
from .forward_rlhf import ForwardStep



def greedy_search(logits, vocab_size=None):
    """ Sample and generate a token.
    Note: logits has the dimension [b, v] where b is the batch size
          and v is the vocabulary size.
    If vocab_size is provided, we will make sure the sample that is
    generated is in [0, vocab-size). This will avoid out of vocabulary
    generations due to padding.
    """

    # Check logits for consistency.
    assert logits.ndim == 2, 'expected the logits to be of [b, v] shape.'
    assert logits.type() == 'torch.cuda.FloatTensor', \
        'input logits should be floats.'

    samples = torch.argmax(logits, dim=-1)

    # If vocab size is provided, make sure the samples are in the range [0, vocab-size).
    if vocab_size:
        samples = torch.clamp(samples, min=0, max=(vocab_size - 1))

    return samples


def generate_tokens_and_return_on_first_stage(
        model, prompts,
        max_answer_seq_len=None,
        pad_token_id=None
        ):
    """Main token generation function.
    Arguments:
        model: no interleaving is supported.
        prompts: prompt tokens extended to be of size [b, prompt_len]
        max_answer_seq_len: The maximum length of generated tokens.
        pad_token_id: The id of the *padding* token.

    Note: Outside of model, other parameters only need to be available on rank 0.

    Outputs:
        tokens: prompt and generated tokens. size: [b, :]
    """

    # Make sure input params are avaialble to all ranks
    values = [max_answer_seq_len, pad_token_id]
    values_float_tensor = broadcast_float_list(len(values), float_list=values)
    max_answer_seq_len = int(values_float_tensor[0].item())
    pad_token_id = int(values_float_tensor[1].item())

    ############ broadcast prompts to all ranks ###########
    sizes_list = None
    prompts_tokens = None
    if torch.distributed.get_rank() == 0:
        assert prompts is not None
        # We need the sizes of these tensors for the boradcast
        sizes_list = [prompts.size(0), prompts.size(1)] # [bsz, seq_len]

    # First, broadcast the sizes.
    sizes_tensor = broadcast_int_list(2, int_list=sizes_list, rank=0)

    # Now that we have the sizes, we can boradcast the tokens
    sizes = sizes_tensor.tolist()
    prompts_tokens = broadcast_tensor(sizes, torch.int64, tensor=prompts, rank=0)

    batch_size, prompt_length = prompts_tokens.size()
    max_sequence_length = prompt_length + max_answer_seq_len

    # Prompt tokens extended to be of size [b, max_sequence_length]
    tokens = F.pad(prompts_tokens, (0, max_answer_seq_len), mode='constant', value=pad_token_id)

    # Forward step
    forward_step = ForwardStep(model, batch_size, max_sequence_length)

    # Run infernece
    tokenizer = get_tokenizer()
    with torch.no_grad():
        attention_mask, position_ids = get_attention_mask_and_position_ids(tokens, pad_token_id=pad_token_id)
        prev_context_length = 0
        for context_length in range(prompt_length, max_sequence_length):

            # Pick the slice that we need to pass through the network.
            tokens2use = tokens[:, prev_context_length:context_length]
            positions2use = position_ids[:, prev_context_length:context_length]
            attention_mask2use = attention_mask[
                ..., prev_context_length:context_length, :context_length]

            # logits will be meanigful only in the last pipeline stage.
            logits = forward_step(tokens2use, positions2use, attention_mask2use)

            if mpu.is_pipeline_last_stage():
                # Always the last stage should have an output.
                assert logits is not None

                # Sample.
                last_token_logits = logits[:, -1, :].contiguous()
                last_token_logits = tensor_parallel.gather_from_tensor_model_parallel_region(last_token_logits)
                new_sample = greedy_search(last_token_logits, vocab_size=tokenizer.vocab_size)

                # Update the tokens
                tokens[:, context_length] = new_sample

            # Update the tokens on the first stage so the next input to
            # the network is correct.
            copy_from_last_to_first_pipeline_stage(batch_size, torch.int64,
                                                   tokens[:, context_length])

            # Update the context length for the next token generation.
            prev_context_length = context_length

    return tokens


def get_attention_mask_and_position_ids(data, pad_token_id=None):
    """Build attention_mask and position_ids for left to right model."""

    # Extract batch size and sequence length.
    micro_batch_size, seq_length = data.size()

    # Position ids.
    position_ids = torch.arange(seq_length, dtype=torch.long,
                                device=data.device)
    position_ids = position_ids.unsqueeze(0).expand_as(data)

    # Attention mask.
    attention_mask = torch.tril(torch.ones(
        (micro_batch_size, seq_length, seq_length), device=data.device)).view(
            micro_batch_size, 1, seq_length, seq_length)

    if pad_token_id is not None:
        # 针对 left_padding 部分更新 attention_mask 和 position_ids
        for b in range(micro_batch_size):
            num_left_padding = 0
            while num_left_padding < len(data[b]) and data[b][num_left_padding] == pad_token_id:
                num_left_padding += 1

            # 更新 attention_mask
            attention_mask[b, :, :, :num_left_padding] = 0

            # 更新 position_ids
            position_ids[b, :num_left_padding] = 1  
            value = 0
            index = num_left_padding
            while index < seq_length:
                position_ids[b, index] = value
                value += 1
                index += 1

    # Convert attention mask to binary:
    attention_mask = (attention_mask < 0.5)

    return attention_mask, position_ids
