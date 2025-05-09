# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Rollout with huggingface models.
TODO: refactor this class. Currently, it will hang when using FSDP HybridShard. We should actually create a single GPU model.
Then, get full state_dict and bind the state_dict to the single GPU model. Then, use the single GPU model to perform generation.
"""
import contextlib
import torch
import torch.distributed
from tensordict import TensorDict
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl import DataProto
from verl.utils.torch_functional import get_eos_mask
from .base import BaseRollout

from transformers import GenerationConfig

__all__ = ['HFRollout']


class HFRollout(BaseRollout):

    def __init__(self, module: nn.Module, config):
        super().__init__()
        self.config = config
        self.module = module

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        batch_size = prompts.batch.batch_size[0]
        num_chunks = max(batch_size // self.config.get('micro_batch_size', batch_size), 1)
        batch_prompts = prompts.chunk(chunks=num_chunks)
        output = [self._generate_minibatch(p) for p in batch_prompts]
        output = DataProto.concat(output)
        return output

    @torch.no_grad()
    def _generate_minibatch(self, prompts: DataProto) -> DataProto:
        idx = prompts.batch['input_ids']  # (bs, prompt_length)
        attention_mask = prompts.batch['attention_mask']  # left-padded attention_mask
        position_ids = prompts.batch['position_ids']

        # used to construct attention_mask
        eos_token_id = prompts.meta_info['eos_token_id']
        pad_token_id = prompts.meta_info['pad_token_id']

        batch_size = idx.size(0)
        prompt_length = idx.size(1)

        self.module.eval()
        param_ctx = contextlib.nullcontext()

        # make sampling args can be overriden by inputs
        do_sample = prompts.meta_info.get('do_sample', self.config.do_sample)
        response_length = prompts.meta_info.get('response_length', self.config.response_length)
        top_p = prompts.meta_info.get('top_p', self.config.get('top_p', 1.0))
        top_k = prompts.meta_info.get('top_k', self.config.get('top_k', 0))

        if top_k is None:
            top_k = 0
        top_k = max(0, top_k)  # to be compatible with vllm

        temperature = prompts.meta_info.get('temperature', self.config.temperature)

        generation_config = GenerationConfig(temperature=temperature, top_p=top_p, top_k=top_k)

        if isinstance(self.module, FSDP):
            # recurse need to set to False according to https://github.com/pytorch/pytorch/issues/100069
            param_ctx = FSDP.summon_full_params(self.module, writeback=False, recurse=False)
        with param_ctx:
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                output = self.module.generate(
                    input_ids=idx,
                    attention_mask=attention_mask,
                    do_sample=do_sample,
                    max_new_tokens=response_length,
                    # max_length=max_length,
                    eos_token_id=eos_token_id,
                    pad_token_id=pad_token_id,
                    generation_config=generation_config,
                    # renormalize_logits=True,
                    output_scores=False,  # this is potentially very large
                    return_dict_in_generate=True,
                    use_cache=True)
        # TODO: filter out the seq with no answers like ds-chat
        seq = output.sequences

        # huggingface generate will stop generating when all the batch reaches [EOS].
        # We have to pad to response_length
        sequence_length = prompt_length + self.config.response_length
        delta_length = sequence_length - seq.shape[1]

        if delta_length > 0:
            delta_tokens = torch.ones(size=(batch_size, delta_length), device=seq.device, dtype=seq.dtype)
            delta_tokens = pad_token_id * delta_tokens
            seq = torch.cat((seq, delta_tokens), dim=1)

        assert seq.shape[1] == sequence_length

        prompt = seq[:, :prompt_length]  # (bs, prompt_length)
        response = seq[:, prompt_length:]  # (bs, response_length)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(batch_size, 1)

        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)

        response_attention_mask = get_eos_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        batch = TensorDict(
            {
                'prompts': prompt,
                'responses': response,
                'input_ids': seq,
                'attention_mask': attention_mask,
                'position_ids': position_ids
            },
            batch_size=batch_size)

        # empty cache before compute old_log_prob
        torch.cuda.empty_cache()

        self.module.train()
        return DataProto(batch=batch)
