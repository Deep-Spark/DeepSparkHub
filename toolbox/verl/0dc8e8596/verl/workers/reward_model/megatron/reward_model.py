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
Megatron Reward Model.
"""

from tensordict import TensorDict
from verl import DataProto
import torch
import torch.distributed

from verl.utils.torch_functional import pad_sequence_to_length
from verl.utils.megatron.pipeline_parallel import (compute_transformers_input_shapes, make_batch_generator)
from verl import DataProto
from verl.utils.torch_functional import broadcast_dict_tensor, split_dict_tensor_into_batches
from verl.workers.reward_model.base import BasePPORewardModel
from megatron.core import parallel_state as mpu
from megatron.core.pipeline_parallel import get_forward_backward_func


class MegatronRewardModel(BasePPORewardModel):

    def __init__(self,
                 config,
                 model_config,
                 reward_model_module: torch.nn.ModuleList,
                 megatron_config,
                 sft_tokenizer=None,
                 rm_tokenizer=None):
        self.config = config
        self.reward_model_module = reward_model_module
        self.megatron_config = megatron_config
        self.model_config = model_config
        self.device = 'cuda'
        self.sft_tokenizer = sft_tokenizer
        self.rm_tokenizer = rm_tokenizer
        self.use_different_tokenizer = rm_tokenizer is not None

        if self.config.param_offload:
            self.offload_params_to_cpu()

    def re_encode_by_rm_tokenizer(self, data: DataProto) -> DataProto:
        assert self.use_different_tokenizer, 're-encode need rm tokenizer not be None!'
        # need to use rm tokenizer to re-generate input_ids, attention_mask and position_ids
        # 1. remove pad for each sequence
        # 2. decode by sft_tokenizer, remove sft system prompts
        # 3. encode by rm_tokenizer with rm system prompts, get rm_input_ids
        # 4. generate attention_mask and position_ids
        input_ids = data.batch['input_ids']  # (bs, seq_len)
        attention_mask = data.batch['attention_mask']
        position_ids = data.batch['position_ids']
        ori_values = {'input_ids': input_ids, 'attention_mask': attention_mask, 'position_ids': position_ids}
        ori_bs, ori_seqlen = input_ids.size(0), input_ids.size(1)
        input_ids_for_rm = []
        attention_mask_for_rm = []
        position_ids_for_rm = []
        print_decode = True
        ori_seqlen = ori_seqlen + 128
        for id, mask in zip(input_ids, attention_mask):
            # 1. remove pad for each sequence
            non_zero_indices = torch.nonzero(mask).view(-1)
            begin_pos, end_pos = non_zero_indices[0].item(), non_zero_indices[-1].item()
            valid_id = id[begin_pos:end_pos + 1]
            # 2. decode by sft_tokenizer, remove sft system prompts
            decode_result = self.sft_tokenizer.decode(valid_id)
            # workaround
            decode_with_rm_chat = decode_result.replace("<|user|>\n", "[INST] ").replace(
                "</s>\n<|assistant|>\n", " [/INST]").replace("</s> \n<|assistant|>\n", " [/INST]") + "</s>"
            if print_decode and torch.distributed.get_rank() == 0:
                # only print first decode result
                print(f'device {torch.cuda.current_device()}: sft decode result:\n{decode_result}\n \
                        \ndevice {torch.cuda.current_device()}: sft decode result with rm chat template:\n{decode_with_rm_chat}\n\n'
                     )
                print_decode = False
            # 3. encode by rm_tokenizer
            rm_input_ids = self.rm_tokenizer(decode_with_rm_chat,
                                             return_tensors='pt')['input_ids'][0].to(input_ids.device)
            # 4. generate attention_mask and position_ids
            rm_attention_mask = torch.ones_like(rm_input_ids, device=input_ids.device)
            cur_seqlen = rm_input_ids.shape[-1]
            # NOTE(gh): the later reward compute will process the shape (bs, seqlen_pad_128)
            if cur_seqlen > ori_seqlen:
                print(f'warninig: rm encode seqlen {cur_seqlen} > sft encode seqlen {ori_seqlen}')
                rm_input_ids = rm_input_ids[:ori_seqlen]
                rm_attention_mask = rm_attention_mask[:ori_seqlen]
            else:
                # right padding
                rm_input_ids = pad_sequence_to_length(rm_input_ids, ori_seqlen, self.rm_tokenizer.pad_token_id)
                rm_attention_mask = pad_sequence_to_length(rm_attention_mask, ori_seqlen, 0)
            rm_position_ids = torch.arange(0, ori_seqlen, device=input_ids.device)
            input_ids_for_rm.append(torch.unsqueeze(rm_input_ids, dim=0))
            attention_mask_for_rm.append(torch.unsqueeze(rm_attention_mask, dim=0))
            position_ids_for_rm.append(torch.unsqueeze(rm_position_ids, dim=0))
        input_ids_for_rm = torch.cat(input_ids_for_rm, dim=0)
        attention_mask_for_rm = torch.cat(attention_mask_for_rm, dim=0)
        position_ids_for_rm = torch.cat(position_ids_for_rm, dim=0)

        # (bs, seqlen) will not change, but input_ids, attention_mask and position_ids will change
        # NOTE(gh): need to replace into origin values after compute reward!
        data.batch['input_ids'] = input_ids_for_rm
        data.batch['attention_mask'] = attention_mask_for_rm
        data.batch['position_ids'] = position_ids_for_rm

        return data, ori_values

    @torch.no_grad()
    def compute_reward(self, data: DataProto) -> DataProto:
        if self.config.param_offload:
            self.load_params_to_cuda()

        if self.use_different_tokenizer:
            data, ori_values = self.re_encode_by_rm_tokenizer(data)

        input_ids = data.batch['input_ids']  # (bs, seq_len')
        attention_mask = data.batch['attention_mask']
        position_ids = data.batch['position_ids']

        responses = data.batch['responses']
        batch_size = responses.size(0)
        response_length = responses.size(1)

        with torch.no_grad():
            output = self.forward_batch(data)
            if mpu.is_pipeline_last_stage(ignore_virtual=True):
                logits = torch.cat([o['logits'] for o in output], dim=0)
            else:
                logits = torch.empty(
                    (input_ids.shape[0], input_ids.shape[1]),
                    dtype=torch.bfloat16,  # TODO(sgm): check why is bfloat16
                    device=input_ids.device)
            # broadcast across pp ranks
            torch.distributed.broadcast(tensor=logits,
                                        src=mpu.get_pipeline_model_parallel_last_rank(),
                                        group=mpu.get_pipeline_model_parallel_group(),
                                        async_op=False)

        # (bs, seqlen', hidden_size) -> (bs, seqlen', 1) -> (bs, seqlen')
        token_level_rewards = logits
        # find the last token reward
        ends = attention_mask.cumsum(dim=-1).argmax(dim=-1).view(-1, 1)  # (bs, 1)
        rewards = torch.gather(token_level_rewards, dim=1, index=ends)  # (bs, 1)

        if self.use_different_tokenizer:
            data.batch.update(ori_values)
            input_ids = ori_values['input_ids']
            attention_mask = ori_values['attention_mask']
            position_ids = ori_values['position_ids']

        token_level_rewards = rewards.expand(attention_mask.shape[0], attention_mask.shape[1])  # (bs, ori_seqlen)

        # assign last valid token reward to ori position
        eos_mask_idx = torch.argmax(position_ids * attention_mask, dim=-1)  # (bs,)
        eos_mask = torch.zeros_like(attention_mask)
        eos_mask[torch.arange(batch_size), eos_mask_idx] = 1.

        token_level_rewards = token_level_rewards * eos_mask
        token_level_rewards = token_level_rewards[:, -response_length:]

        if self.config.param_offload:
            self.offload_params_to_cpu()
        else:
            # add empty cache after each compute
            torch.cuda.empty_cache()

        batch = TensorDict({'rm_scores': token_level_rewards}, batch_size=input_ids.shape[0])

        return DataProto(batch=batch)

    def forward_batch(self, data: DataProto):
        """
        We assume:
        - The model takes input: (input_ids, attention_mask, position_ids). No rmpad for the input
        - The communication shape is (total_nnz_pad_to_sp // tp_size, 1, hidden_size) if sequence parallel is enabled
        """
        # broadcast from last pp rank to all other pp ranks
        # TODO: actually, we just need to control the sampling order.
        data.batch = data.batch.contiguous()
        broadcast_dict_tensor(data.batch,
                              src=mpu.get_pipeline_model_parallel_last_rank(),
                              group=mpu.get_pipeline_model_parallel_group())

        # split into micro-batches
        if self.config is not None and 'ppo_micro_batch_size_per_gpu' in self.config:
            infer_batch_size = self.config.ppo_micro_batch_size_per_gpu
        else:
            infer_batch_size = data.batch.batch_size[0]

        data.batch['attention_mask'] = data.batch['attention_mask'].to(bool)
        batches = split_dict_tensor_into_batches(data.batch, batch_size=infer_batch_size)
        n_micro_batch = len(batches)
        seq_len = batches[0]['input_ids'].shape[1]

        # compute input shapes for pp stages
        input_shapes = compute_transformers_input_shapes(
            batches,
            meta_info={
                'sequence_parallel': self.megatron_config.sequence_parallel,
                'hidden_size': self.model_config.hidden_size
            })
        # compute input shapes for pp stages
        forward_backward_func = get_forward_backward_func()

        def loss_func(output):
            return 1., {'logits': output.logits}

        def forward_step(batch_iter, model):
            batch = next(batch_iter)
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            position_ids = batch['position_ids']
            output = model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids)
            return output, loss_func

        # batch should be a list of batches inside micro-batches
        batch_generator = make_batch_generator(batches, vpp_size=len(self.reward_model_module))

        # TODO: we may use the new schedule instead
        # for flash-attn: (seq_len, batch_size, hidden_size) = (mbs*seq_len, 1, hidden_size)
        if mpu.get_pipeline_model_parallel_world_size() > 1:
            losses_reduced = forward_backward_func(
                forward_step_func=forward_step,
                data_iterator=batch_generator,
                model=self.reward_model_module,
                num_microbatches=n_micro_batch,
                input_shapes=input_shapes,  # must set for flash-attn sequence packing
                seq_length=infer_batch_size * seq_len,  # no use when input_shapes was set
                hidden_size=self.model_config.hidden_size,  # no use when input_shapes was set
                micro_batch_size=1,  # no use when input_shapes was set
                forward_only=True,
            )
        else:
            losses_reduced = forward_backward_func(
                forward_step_func=forward_step,
                data_iterator=batch_generator,
                model=self.reward_model_module,
                num_microbatches=n_micro_batch,
                seq_length=infer_batch_size * seq_len,  # in use for pp = 1
                hidden_size=self.model_config.hidden_size,  # in use for pp = 1
                micro_batch_size=1,  # in use for pp = 1
                forward_only=True,
            )
        # loss_reduces contains the stats returned from loss_func

        return losses_reduced

    def offload_params_to_cpu(self):
        if self.device == 'cuda':
            for reward_model_module in self.reward_model_module:
                for name, param in reward_model_module.named_parameters():
                    param.data = param.data.to('cpu', non_blocking=True)
            self.device = 'cpu'
            torch.cuda.empty_cache()

    def load_params_to_cuda(self):
        if self.device == 'cpu':
            for reward_model_module in self.reward_model_module:
                for name, param in reward_model_module.named_parameters():
                    param.data = param.data.to(torch.cuda.current_device(), non_blocking=True)
            self.device = 'cuda'
