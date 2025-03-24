"""Pretrain utilities."""

import dataclasses
from datetime import datetime
import gc
from functools import partial
import math
import logging
import sys
from ..log_handler import CustomHandler
# Make default logging level INFO, but filter out all log messages not from MCore.
logging.basicConfig(handlers=[CustomHandler()], level=logging.INFO)
import time
import os
# The earliest we can measure the start time.
_TRAIN_START_TIME = time.time()
import torch
import copy
import torch.nn.functional as F

from megatron.core import mpu, tensor_parallel
from megatron.core.utils import get_model_config
from megatron.legacy.model import Float16Module, GPTModel
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.distributed import finalize_model_grads
from megatron.core.enums import ModelType
from megatron.core.optimizer import get_megatron_optimizer, OptimizerConfig
from megatron.training.optimizer_param_scheduler import OptimizerParamScheduler
from megatron.legacy.data.data_samplers import build_pretraining_data_loader
from megatron.core.num_microbatches_calculator import (
    get_current_global_batch_size,
    get_num_microbatches,
    update_num_microbatches)
from megatron.training.utils import (
    calc_params_l2_norm,
    check_adlr_autoresume_termination,
    is_last_rank,
    print_rank_0,
    print_rank_last,
    report_memory,
    unwrap_model,
    append_to_progress_log,
    get_ltor_masks_and_position_ids,
    get_batch_on_this_tp_rank,
    get_batch_on_this_cp_rank,
    average_losses_across_data_parallel_group
)
from megatron.training.global_vars import (
    get_args,
    get_rlhf_args,
    set_rlhf_args,
    set_args,
    get_tokenizer,
    get_signal_handler,
    get_timers,
    get_tensorboard_writer,
    get_wandb_writer,
    get_one_logger)
from megatron.training import one_logger_utils
from megatron.training.training import (
    print_datetime,
    num_floating_point_operations,
    update_train_iters,
    build_train_valid_test_data_iterators,
    get_optimizer_param_scheduler,
    build_train_valid_test_datasets,
    save_checkpoint_and_time
)
from megatronspeed.rlhf.schedules_rlhf import get_forward_backward_func
from megatronspeed.rlhf.initialize_rlhf import initialize_megatron, write_args_to_tensorboard
from megatronspeed.rlhf.checkpointing_rlhf import load_state_dict_into_model, load_state_dict, save_checkpoint, set_args_from_state_dict
from megatronspeed.rlhf.generation.generation_rlhf import generate_tokens_and_return_on_first_stage, get_attention_mask_and_position_ids
from megatronspeed.rlhf.generation.communication_rlhf import broadcast_from_last_pipeline_stage




def print_datetime(string):
    """Note that this call will sync across all ranks."""
    torch.distributed.barrier()
    time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print_rank_0('[' + string + '] datetime: {} '.format(time_str))


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


def gather_log_probs(logits, labels):
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1))
    return log_probs_labels.squeeze(-1)


class RLHFPPOTrainer():
    def __init__(self,
                 train_valid_test_dataset_provider,
                 model_provider,
                 model_type,
                 forward_step_func=None,
                 process_non_loss_data_func=None,
                 extra_args_provider=None,
                 args_defaults={}):
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
        self.model_provider = model_provider
        self.model_type = model_type

        # Those value can be changed
        self.kl_ctl = 0.1
        self.clip_reward_value = 5
        self.cliprange = 0.2
        self.cliprange_value = 0.2
        self.gamma = 1.0
        self.lam = 0.95

        # Initalize and get arguments, timers, and Tensorboard writer.
        initialize_megatron(extra_args_provider=extra_args_provider,
                            args_defaults=args_defaults)

        # Adjust the startup time so it reflects the largest value.
        # This will be closer to what scheduler will see
        global _TRAIN_START_TIME
        start_time_tensor = torch.cuda.DoubleTensor([_TRAIN_START_TIME])
        torch.distributed.all_reduce(start_time_tensor,
                                    op=torch.distributed.ReduceOp.MIN)
        _TRAIN_START_TIME = start_time_tensor.item()
        print_rank_0('time to initialize megatron (seconds): {:.3f}'.format(
            time.time() - _TRAIN_START_TIME))
        print_datetime('after megatron is initialized')

        # separate args between actor/critic model
        self.args = get_args()
        # reset seq_length argument
        self.max_seq_len = self.args.max_prompt_seq_len + self.args.decoder_seq_length
        if self.args.seq_length != self.max_seq_len :
            setattr(self.args, "seq_length", self.max_seq_len)
            set_args(self.args)
        # copy args to rlhf_args, which will be updated during loading model
        self.rlhf_args = copy.deepcopy(self.args)
        set_rlhf_args(self.rlhf_args)
        # reset num_layers_per_stage argument
        if self.args.num_layers_per_stage is not None and self.args.num_layers != sum(self.args.num_layers_per_stage):
            setattr(self.args, "num_layers_per_stage", None)
            set_args(self.args)

        self.timers = get_timers()
        self.tokenizer = get_tokenizer()
        self.pad_token_id = 0

        # Create Actor/Reference Model
        self.actor_model, self.actor_optimizer, self.actor_opt_param_scheduler \
            = self.init_rlhf_model(model_prefix="actor", rlhf_training=False)
        self.actor_config = get_model_config(self.actor_model[0])
        self.reference_model, _, _ = self.init_rlhf_model(model_prefix="reference", rlhf_training=False)

        # Create Critic/Reward Model
        self.critic_model, self.critic_optimizer, self.critic_opt_param_scheduler \
            = self.init_rlhf_model(model_prefix="critic", rlhf_training=True)
        self.critic_config = get_model_config(self.critic_model[0])
        self.reward_model, _, _ = self.init_rlhf_model(model_prefix="reward", rlhf_training=True)

        print_datetime('after actor/reference/critic/reward model is built')

        # Data stuff.
        self.timers('train/valid/test-data-iterators-setup', log_level=0).start(barrier=True)
        self.train_data_iterator, self.valid_data_iterator, \
            self.test_data_iterator = build_train_valid_test_data_iterators(train_valid_test_dataset_provider)
        self.timers('train/valid/test-data-iterators-setup').stop()
        self.timers.log(['train/valid/test-data-iterators-setup'], barrier=True)

        # Get the batch.
        data_iterator = self.train_data_iterator
        if isinstance(data_iterator, list):
            assert (
                len(data_iterator) == 1
            ), "non-pipeline-parallel schedule does not support model chunking"
            data_iterator = data_iterator[0]
        if self.args.do_train and self.args.train_iters > 0:
            iteration = self.train(data_iterator=data_iterator)
        

    def init_rlhf_model(self, model_prefix=None, rlhf_training=False):
        """Setup rlhf actor/critic model"""
        if rlhf_training:
            args = get_rlhf_args()
        else:
            args = get_args()

        if model_prefix in {"actor", "reference"}:
            ckpt_dir = getattr(args, "actor_model_name_or_path")
        elif model_prefix in {"critic", "reward"}:
            ckpt_dir = getattr(args, "critic_model_name_or_path")
            assert rlhf_training, "Init model should be critic or reward when rlhf_training is True"
        else:
            raise Exception(f'model_prefix should be in [actor|reference|critic|reward].')

        state_dict = load_state_dict(ckpt_dir)
        set_args_from_state_dict(args, state_dict, rlhf_training=rlhf_training)

        args.encoder_num_layers = args.num_layers
        if rlhf_training:
            set_rlhf_args(args)
        else:
            set_args(args)

        # Model
        model = get_model(self.model_provider, self.model_type,
                          rlhf_training=rlhf_training)

        # Optimizer
        optimizer, opt_param_scheduler = None, None
        if model_prefix in {"actor", "critic"}:
            kwargs = {}
            for f in dataclasses.fields(OptimizerConfig):
                if hasattr(args, f.name):
                    kwargs[f.name] = getattr(args, f.name)
            config = OptimizerConfig(**kwargs)
            config.timers = self.timers
            config.lr = getattr(args, f"{model_prefix}_learning_rate")
            config.weight_decay = getattr(args, f"{model_prefix}_weight_decay")
            args.lr = config.lr
            args.weight_decay = config.weight_decay
            optimizer = get_megatron_optimizer(config, model)
            opt_param_scheduler = get_optimizer_param_scheduler(optimizer)

        if ckpt_dir is not None:
            self.timers(f'load {model_prefix} model', log_level=0).start(barrier=True)
            load_state_dict_into_model(model, state_dict)
            self.timers(f'load {model_prefix} model').stop(barrier=True)
            self.timers.log([f'load {model_prefix} model'])
        else:
            raise Exception(f'{model_prefix}_model_name_or_path should be provided.')

        # We only support local DDP with multiple micro-batches.
        if len(model) > 1 or mpu.get_pipeline_model_parallel_world_size() > 1:
            assert args.DDP_impl == 'local'

        return model, optimizer, opt_param_scheduler


    def generate_experience(self, prompts):
        ''' RLHF 第一阶段四个模型推理 '''

        # 将 actor/reference/critic/reward 转为 eval 模式
        self.set_eval()

        # Actor model 输入 max_prompt_seq_len 生成 max_answer_seq_len
        # 返回 max_prompt_seq_len + max_answer_seq_len sequence
        seq = self.generate_sequence(prompts)
        attention_mask = seq.not_equal(self.pad_token_id).long()

        # broadcast prompts|seq|attention_mask
        size = (self.args.micro_batch_size, self.args.max_prompt_seq_len)
        prompts = broadcast_from_last_pipeline_stage(size, torch.int64, prompts)
        size = (self.args.micro_batch_size, self.args.seq_length)
        seq = broadcast_from_last_pipeline_stage(size, torch.int64, seq)
        attention_mask = broadcast_from_last_pipeline_stage(size, torch.int64, attention_mask)

        size = (self.args.micro_batch_size, self.args.seq_length, self.args.padded_vocab_size)

        self.micro_batch_size = self.args.rlhf_train_mbs
        self.num_microbatches = seq.shape[0] // self.micro_batch_size
        assert seq.shape[0] % self.micro_batch_size == 0

        # 1. actor model 生成 logits
        seq_tmp = seq.clone().detach()
        with torch.no_grad():
            output_data = self.forward_backward_func(
                forward_step_func=self.forward_func,
                prompts=seq_tmp,
                model=self.actor_model,
                num_microbatches=self.num_microbatches,
                seq_length=self.args.seq_length,
                micro_batch_size=self.micro_batch_size,
                decoder_seq_length=1,
                forward_only=True,
                collect_non_loss_data=True,
                model_prefix='actor')
            if mpu.is_pipeline_last_stage():
                logits = torch.cat(output_data, dim=0) # [b, seq_len, v]
            else:
                logits = None
            if self.args.empty_unused_memory_level >= 1:
                if mpu.is_pipeline_last_stage():
                    logits_tmp = logits.clone().detach() if logits is not None else None
                    logits_tmp = tensor_parallel.gather_from_tensor_model_parallel_region(logits_tmp)
                    del seq_tmp, output_data, logits
                    torch.cuda.empty_cache()
                    logprobs = gather_log_probs(logits_tmp, seq[:, self.args.max_prompt_seq_len:]).clone().detach()
                    del logits_tmp
                    torch.cuda.empty_cache()
                else:
                    logprobs = None
            else:
                if mpu.is_pipeline_last_stage():
                    logits_tmp = logits.contiguous()
                    logits_tmp = tensor_parallel.gather_from_tensor_model_parallel_region(logits_tmp)
                    logprobs = gather_log_probs(logits_tmp, seq[:, self.args.max_prompt_seq_len:]).clone().detach()
                else:
                    logprobs = None
            size = (self.args.micro_batch_size, self.args.decoder_seq_length)
            logprobs = broadcast_from_last_pipeline_stage(size, torch.float32, logprobs)

        # 2. reference model 生成 ref_logits
        seq_tmp = seq.clone().detach()
        with torch.no_grad():
            output_data = self.forward_backward_func(
                forward_step_func=self.forward_func,
                prompts=seq_tmp,
                model=self.reference_model,
                num_microbatches=self.num_microbatches,
                seq_length=self.args.seq_length,
                micro_batch_size=self.micro_batch_size,
                decoder_seq_length=1,
                forward_only=True,
                collect_non_loss_data=True,
                model_prefix='reference')
            if mpu.is_pipeline_last_stage():
                ref_logits = torch.cat(output_data, dim=0) # [b, seq_len, v]
            else:
                ref_logits = None
            if self.args.empty_unused_memory_level >= 1:
                if mpu.is_pipeline_last_stage():
                    ref_logits_tmp = ref_logits.clone().detach() if ref_logits is not None else None
                    ref_logits_tmp = tensor_parallel.gather_from_tensor_model_parallel_region(ref_logits_tmp)
                    del seq_tmp, output_data, ref_logits
                    torch.cuda.empty_cache()
                    ref_logprobs = gather_log_probs(ref_logits_tmp, seq[:, self.args.max_prompt_seq_len:]).clone().detach()
                    del ref_logits_tmp
                    torch.cuda.empty_cache()
                else:
                    ref_logprobs = None
            else:
                if mpu.is_pipeline_last_stage():
                    ref_logits_tmp = ref_logits.contiguous()
                    ref_logits_tmp = tensor_parallel.gather_from_tensor_model_parallel_region(ref_logits_tmp)
                    ref_logprobs = gather_log_probs(ref_logits_tmp, seq[:, self.args.max_prompt_seq_len:]).clone().detach()
                else:
                    ref_logprobs = None

            size = (self.args.micro_batch_size, self.args.decoder_seq_length)
            ref_logprobs = broadcast_from_last_pipeline_stage(size, torch.float32, ref_logprobs)

        size = (self.args.micro_batch_size, self.args.decoder_seq_length)
        # 3. critic model 生成 values
        seq_tmp = seq.clone().detach()
        with torch.no_grad():
            output_data = self.forward_backward_func(
                forward_step_func=self.forward_func,
                prompts=seq_tmp,
                model=self.critic_model,
                num_microbatches=self.num_microbatches,
                seq_length=self.args.seq_length,
                micro_batch_size=self.micro_batch_size,
                decoder_seq_length=1,
                forward_only=True,
                collect_non_loss_data=True,
                model_prefix='critic')
            if mpu.is_pipeline_last_stage():
                values_tmp = torch.cat(output_data, dim=0) # [b, seq_len]
            else:
                values_tmp = None
            # values = broadcast_from_last_pipeline_stage(size, torch.float32, values) # [b, decoder_seq_len]
            if self.args.empty_unused_memory_level >= 1:
                if mpu.is_pipeline_last_stage():
                    values = values_tmp[:, self.args.max_prompt_seq_len-1:-1].clone().detach()
                    del seq_tmp, output_data, values_tmp
                    torch.cuda.empty_cache()
                else:
                    values = None
            else:
                if mpu.is_pipeline_last_stage():
                    values = values_tmp[:, self.args.max_prompt_seq_len-1:-1].contiguous()
                else:
                    values = None
            values = broadcast_from_last_pipeline_stage(size, torch.float32, values)

        # 4. reward model 生成 reward_score
        seq_tmp = seq.clone().detach()
        with torch.no_grad():
            output_data = self.forward_backward_func(
                forward_step_func=self.forward_func,
                prompts=seq_tmp,
                model=self.reward_model,
                num_microbatches=self.num_microbatches,
                seq_length=self.args.seq_length,
                micro_batch_size=self.micro_batch_size,
                decoder_seq_length=1,
                forward_only=True,
                collect_non_loss_data=True,
                model_prefix='reward')
            if mpu.is_pipeline_last_stage():
                reward_values_tmp = torch.cat(output_data, dim=0) # [b, seq_len]
            else:
                reward_values_tmp = None
            # reward_values = broadcast_from_last_pipeline_stage(size, torch.float32, reward_values)
            # reward_score = self.postprocess_reward_forward_output(seq, reward_values) # [b]
            if self.args.empty_unused_memory_level >= 1:
                if mpu.is_pipeline_last_stage():
                    reward_values = reward_values_tmp.clone().detach()
                    del seq_tmp, output_data, reward_values_tmp
                    torch.cuda.empty_cache()
                    reward_score = self.postprocess_reward_forward_output(seq, reward_values) # [b]
                else:
                    reward_score = None
            else:
                if mpu.is_pipeline_last_stage():
                    reward_score = self.postprocess_reward_forward_output(seq, reward_values_tmp)
                else:
                    reward_score = None
            size = (self.args.micro_batch_size)
            reward_score = broadcast_from_last_pipeline_stage(size, torch.float32, reward_score)

        # 将 actor/critic 转为 train 模式
        self.set_train()

        # 由于 logits 是输入seq actor_model 下一时刻的输出, 通过错位操作让 logits 和 seq 一一对应, 然后取log
        # 由于 ref_logits 是输入seq reference_model 下一时刻的输出, 通过错位操作让 logits 和 seq 一一对应, 然后取log
        return {
            'prompts': prompts,
            'logprobs': logprobs,
            'ref_logprobs': ref_logprobs,
            'value': values,
            'rewards': reward_score,
            'input_ids': seq,
            "attention_mask": attention_mask
        }


    def generate_sequence(self, prompts):

        model = self.actor_model
        if isinstance(model, list):
            assert len(model) == 1, "non-pipeline-parallel schedule does not support model chunking"
            model = model[0]

        self.timers('generate_sequence',log_level=0).start()
        with torch.no_grad():
            seq = generate_tokens_and_return_on_first_stage(model, prompts,
                                                            max_answer_seq_len=self.args.decoder_seq_length,
                                                            pad_token_id=self.pad_token_id)

            # Empty unused memory
            if self.args.empty_unused_memory_level >= 1:
                torch.cuda.empty_cache()
        self.timers('generate_sequence').stop()

        return seq


    def set_train(self):
        # Set model to the train mode.
        for model_module in self.actor_model:
            model_module.train()
        for model_module in self.critic_model:
            model_module.train()


    def set_eval(self):
        # Set model to evaluation mode which disables dropout.
        for model_module in self.actor_model:
            model_module.eval()
        for model_module in self.reference_model:
            model_module.eval()
        for model_module in self.critic_model:
            model_module.eval()
        for model_module in self.reward_model:
            model_module.eval()


    def postprocess_reward_forward_output(self, tokens, values):
        """postprocess reward model forward output to get reward score.
        Args:
            tokens: reward model input tokens [b, seq_len]
            values: reward model output values [b, seq_len]
        """
        prompt_len, seq_len = self.args.max_prompt_seq_len, self.args.seq_length
        assert prompt_len > 1, "prompt_length must be greater than 1 to help select the end score"

        # Get the end score
        batch_size = values.size(0)
        chosen_end_scores = []
        for i in range(batch_size):
            token, value = tokens[i], values[i]
            c_inds = (token[prompt_len:] == self.pad_token_id).nonzero()
            # here we only use the answer part of the sequence so we do not need to care about the padding at the beginning
            c_ind = c_inds[0].item() + prompt_len if len(c_inds) > 0 else seq_len
            chosen_end_scores.append(value[c_ind - 1])

        return torch.stack(chosen_end_scores)


    def compute_rewards(self, log_probs, ref_log_probs, reward_score):
        '''
            使用 actor/reference 结果的 KL Divergence 来修正 rewards

            log_probs: [bsz, decoder_seq_len] actor_model forward 后处理结果
            ref_log_probs: [bsz, decoder_seq_len] reference_model forward 后处理结果
            reward_score: [bsz] reward_model forward 后处理结果
        '''
        kl_divergence_estimate = -self.kl_ctl * (log_probs - ref_log_probs)
        rewards = kl_divergence_estimate

        reward_clip = torch.clamp(reward_score, -self.clip_reward_value, self.clip_reward_value)
        batch_size = log_probs.shape[0]
        for j in range(batch_size):
            end = self.ends[j]
            rewards[j, :end][-1] += reward_clip[j] # [bsz, decoder_seq_len] 更新 end reward_score

        return rewards


    def get_advantages_and_returns(self, values, rewards):
        # Adopted from https://github.com/CarperAI/trlx/blob/main/trlx/models/modeling_ppo.py#L134
        '''
            计算 advantages 和 returns

            values: [bsz, decoder_seq_len] critic model forward 后处理结果
            rewards: [bsz, decoder_seq_len] KL散度修正后 rewards
        '''
        lastgaelam = 0
        advantages_reversed = []
        length = rewards.size()[-1]
        for t in reversed(range(length)):
            nextvalues = values[:, t + 1] if t < length - 1 else 0.0
            delta = rewards[:, t] + self.gamma * nextvalues - values[:, t]
            lastgaelam = delta + self.gamma * self.lam * lastgaelam
            advantages_reversed.append(lastgaelam)

        advantages = torch.stack(advantages_reversed[::-1], dim=1) # [b, decoder_seq_len]
        returns = advantages + values # [b, decoder_seq_len]

        return advantages.detach(), returns


    def train_rlhf(self, inputs):
        prompts = inputs['prompts']
        log_probs = inputs['logprobs'] # [b, decoder_seq_len]
        ref_log_probs = inputs['ref_logprobs'] # [b, decoder_seq_len]
        reward_score = inputs['rewards'] # [b]
        old_values = inputs['value'] # [b, decoder_seq_len]
        attention_mask = inputs['attention_mask']
        seq = inputs['input_ids']

        # 计算优势和回报
        action_mask = attention_mask[:, self.args.max_prompt_seq_len:] # [b, decoder_seq_len]
        self.ends = action_mask.sum(1) + 1 # [b]
        with torch.no_grad():
            # 计算 KL 散度 和 reward model 修正奖励
            old_rewards = self.compute_rewards(log_probs, ref_log_probs, reward_score) # [b, decoder_seq_len]

            # we need to zero out the reward and value after the end of the conversation
            # otherwise the advantage/return will be wrong
            for i in range(old_rewards.shape[0]):
                old_rewards[i, self.ends[i]:] = 0
                old_values[i, self.ends[i]:] = 0

            advantages, returns = self.get_advantages_and_returns(old_values, old_rewards)

        self.timers('actor-train', log_level=0).start()
        actor_loss, actor_skipped_iter, actor_grad_norm, actor_num_zeros_in_grad = self.train_actor(
            seq, log_probs, advantages, action_mask)
        self.timers('actor-train', log_level=0).stop()

        self.timers('critic-train', log_level=0).start()
        critic_loss, critic_skipped_iter, critic_grad_norm, critic_num_zeros_in_grad = self.train_critic(
            seq, old_values, returns, action_mask)
        self.timers('critic-train', log_level=0).stop()
        
        if mpu.is_pipeline_last_stage():
            return [actor_loss['lm loss'].item(), critic_loss['lm loss'].item()], [actor_skipped_iter, critic_skipped_iter], [actor_grad_norm, critic_grad_norm], \
                [actor_num_zeros_in_grad, critic_num_zeros_in_grad]
        else:
            ## 非最后一个PP stage 保证有个输出，避免后续计算出错
            return [0,0], [actor_skipped_iter, critic_skipped_iter], [actor_grad_norm, critic_grad_norm], \
                [actor_num_zeros_in_grad, critic_num_zeros_in_grad]

            
    def train_actor(self, seq, log_probs, advantages, action_mask):
        ################ actor model 训练 ############
        # Set grad to zero.
        for model_chunk in self.actor_model:
            # If using distributed optimizer, don't zero buffer here; zeroing of buffer is
            # handled automatically by the optimizer after all-gathers finish.
            # Otherwise, zero the buffer.
            model_chunk.zero_grad_buffer()
        self.actor_optimizer.zero_grad()

        actor_loss = self.forward_backward_func(
            forward_step_func=self.actor_forward_backward_func,
            prompts=seq,
            model=self.actor_model,
            num_microbatches=self.num_microbatches,
            seq_length=self.args.seq_length,
            micro_batch_size=self.micro_batch_size,
            decoder_seq_length=1,
            forward_only=False,
            old_log_probs=log_probs,
            advantages=advantages,
            action_mask=action_mask)

        # Empty unused memory.
        if self.args.empty_unused_memory_level >= 1:
            torch.cuda.empty_cache()

        # Update parameters.
        self.timers('optimizer', log_level=1).start(barrier=self.args.barrier_with_L1_time)
        update_successful, grad_norm, num_zeros_in_grad = self.actor_optimizer.step()
        self.timers('optimizer').stop()

        # Update learning rate.
        if update_successful:
            increment = self.num_microbatches * \
                        self.micro_batch_size * \
                        self.args.data_parallel_size
            self.actor_opt_param_scheduler.step(increment=increment)
            skipped_iter = 0
        else:
            skipped_iter = 1

        # Empty unused memory.
        if self.args.empty_unused_memory_level >= 2:
            torch.cuda.empty_cache()
        
        if mpu.is_pipeline_last_stage(ignore_virtual=True):
        # Average loss across microbatches.
            loss_reduced = {}
            for key in actor_loss[0]:
                losses_reduced_for_key = [x[key] for x in actor_loss]
                loss_reduced[key] = sum(losses_reduced_for_key) / len(losses_reduced_for_key)
            return loss_reduced, skipped_iter, grad_norm, num_zeros_in_grad
        return {}, skipped_iter, grad_norm, num_zeros_in_grad

    def train_critic(self, seq, old_values, returns, action_mask):
        ################ critic model 训练 ############
        # Set grad to zero.
        for model_chunk in self.critic_model:
            # If using distributed optimizer, don't zero buffer here; zeroing of buffer is
            # handled automatically by the optimizer after all-gathers finish.
            # Otherwise, zero the buffer.
            model_chunk.zero_grad_buffer()
        self.critic_optimizer.zero_grad()

        critic_loss = self.forward_backward_func(
            forward_step_func=self.critic_forward_backward_func,
            prompts=seq,
            model=self.critic_model,
            num_microbatches=self.num_microbatches,
            seq_length=self.args.seq_length,
            micro_batch_size=self.micro_batch_size,
            decoder_seq_length=1,
            forward_only=False,
            old_values=old_values,
            returns=returns,
            action_mask=action_mask)

        # Empty unused memory.
        if self.args.empty_unused_memory_level >= 1:
            torch.cuda.empty_cache()

        # Update parameters.
        self.timers('optimizer', log_level=1).start(barrier=self.args.barrier_with_L1_time)
        update_successful, grad_norm, num_zeros_in_grad = self.critic_optimizer.step()
        self.timers('optimizer').stop()

        # Update learning rate.
        if update_successful:
            increment = self.num_microbatches * \
                        self.micro_batch_size * \
                        self.args.data_parallel_size
            self.critic_opt_param_scheduler.step(increment=increment)
            skipped_iter = 0
        else:
            skipped_iter = 1

        # Empty unused memory.
        if self.args.empty_unused_memory_level >= 2:
            torch.cuda.empty_cache()
        
        if mpu.is_pipeline_last_stage(ignore_virtual=True):
        # Average loss across microbatches.
            loss_reduced = {}
            for key in critic_loss[0]:
                losses_reduced_for_key = [x[key] for x in critic_loss]
                loss_reduced[key] = sum(losses_reduced_for_key) / len(losses_reduced_for_key)
            return loss_reduced, skipped_iter, grad_norm, num_zeros_in_grad
        return {}, skipped_iter, grad_norm, num_zeros_in_grad

    def forward_func(self, tokens, model, model_prefix):
        """Forward Function.

        Args:
            tokens : Input Tokens
            model (GPTModel): The GPT Model
        """

        attention_mask, position_ids = get_attention_mask_and_position_ids(tokens, pad_token_id=self.pad_token_id)

        output_tensor = model(tokens, position_ids, attention_mask, parallel_output=False)

        ## 将一阶段模型前推时需要切分的prompt length长度提前到此处，返回更小的tensor，有利于优化大batch size
        if mpu.is_pipeline_last_stage() and model_prefix in ['actor', 'reference']:
            output_tensor = output_tensor[:, self.args.max_prompt_seq_len-1:-1, :]
        return output_tensor, None


    def actor_forward_backward_func(self, tokens, model, old_log_probs, advantages, action_mask):
        """Forward Function.

        Args:
            tokens (Tensor): Input Tokens
            model (GPTModel): The GPT Model
        """

        attention_mask, position_ids = get_attention_mask_and_position_ids(tokens, pad_token_id=self.pad_token_id)

        output_tensor = model(tokens, position_ids, attention_mask, parallel_output=False)

        return output_tensor, partial(self.actor_loss_func, tokens, old_log_probs, advantages, action_mask)


    def actor_loss_fn(self, logprobs, old_logprobs, advantages, mask):
        log_ratio = (logprobs - old_logprobs) * mask
        ratio = torch.exp(log_ratio)
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(ratio, 1.0-self.cliprange, 1.0+self.cliprange)
        pg_loss = torch.sum(torch.max(pg_loss1, pg_loss2) * mask) / mask.sum()
        return pg_loss


    def actor_loss_func(self, tokens, old_log_probs, advantages, action_mask, output_tensor):
        """Loss function.

        Args:
            tokens (Tensor): [b, s]
            old_log_probs (Tensor): [b, decoder_seq_length]
            advantages (Tensor): [b, decoder_seq_length]
            action_mask (Tensor): [b, decoder_seq_length]
            output_tensor (Tensor): [b, s, v]
        """
        output_tensor = output_tensor[:, self.args.max_prompt_seq_len-1:-1, :]
        output_tensor = tensor_parallel.gather_from_tensor_model_parallel_region(output_tensor)
        actor_log_prob = gather_log_probs(output_tensor,
                                          tokens[:, self.args.max_prompt_seq_len:]) # [b, decoder_seq_length]
        actor_loss = self.actor_loss_fn(actor_log_prob, old_log_probs, advantages, action_mask)

        # Check individual rank losses are not NaN prior to DP all-reduce.
        if self.args.check_for_nan_in_loss_and_grad:
            global_rank = torch.distributed.get_rank()
            assert not actor_loss.isnan(), (
                f'Rank {global_rank}: found NaN in local forward loss calculation. '
                f'Device: {torch.cuda.current_device()}, node: {os.uname()[1]}'
            )

        # Reduce loss for logging.
        averaged_loss = average_losses_across_data_parallel_group([actor_loss])

        return actor_loss, {'lm loss': averaged_loss[0]}

    def critic_forward_backward_func(self, tokens, model: GPTModel, old_values=None, returns=None, action_mask=None):
        """Forward Function.
        Args:
            tokens : Input Tokens
            model (GPTModel): The GPT Model
        """

        attention_mask, position_ids = get_attention_mask_and_position_ids(tokens, pad_token_id=self.pad_token_id)

        output_tensor = model(tokens, position_ids, attention_mask, parallel_output=False)

        return output_tensor, partial(self.critic_loss_func, old_values, returns, action_mask)

    def critic_loss_func(self, old_values, returns, action_mask, output_tensor):
        """Loss function.

        Args:
            old_values (Tensor): [b, decoder_seq_length]
            returns (Tensor): [b, decoder_seq_length]
            action_mask (Tensor): [b, decoder_seq_length]
            output_tensor (Tensor): [b, s]
        """
        critic_loss = self.critic_loss_fn(values=output_tensor[:, self.args.max_prompt_seq_len-1:-1],
                                          old_values=old_values,
                                          returns=returns,
                                          action_mask=action_mask)

        # Check individual rank losses are not NaN prior to DP all-reduce.
        if self.args.check_for_nan_in_loss_and_grad:
            global_rank = torch.distributed.get_rank()
            assert not critic_loss.isnan(), (
                f'Rank {global_rank}: found NaN in local forward loss calculation. '
                f'Device: {torch.cuda.current_device()}, node: {os.uname()[1]}'
            )

        # Reduce loss for logging.
        averaged_loss = average_losses_across_data_parallel_group([critic_loss])

        return critic_loss, {'lm loss': averaged_loss[0]}

    def critic_loss_fn(self, values, old_values, returns, action_mask):
        ## values loss
        # clip 防止训偏
        # 取较大的那一项 有利于稳定训练。 梯度较大 更新方向更明确
        
        values_clipped = torch.clamp(
            values,
            old_values - self.cliprange_value,
            old_values + self.cliprange_value,
        )
        loss1 = (values - returns)**2
        loss2 = (values_clipped - returns)**2
        loss = 0.5 * torch.sum(
            torch.max(loss1, loss2) * action_mask) / action_mask.sum()
        return loss
    
    def train(self, data_iterator):
        """Train the model function."""
        
        # Iterations.
        iteration = self.args.iteration

        while iteration < self.args.train_iters:
            self.args.curr_iteration = iteration
            self.timers('end-to-end', log_level=0).start()
            prompts, labels, loss_mask, attention_mask, position_ids = get_batch(
                    data_iterator)
            self.forward_backward_func = get_forward_backward_func()

            # 第一阶段推理
            out = self.generate_experience(prompts)

            # 第二阶段训练
            self.timers('train-time', log_level=0).start()
            loss_sum, skipped_iter_sum = [0,0], [0,0]
            average_reward = 0
            total_step = 0
            for ppo_ep in range(self.args.ppo_epoches):
                # 后续若有多个数据需要添加遍历循环
                loss, skipped_iter, grad_norm, num_zeros_in_grad = self.train_rlhf(out)

                average_reward += out["rewards"].mean()
                total_step += 1

                loss_sum = [loss_sum[k]+loss[k] for k in range(2)]
                skipped_iter_sum = [skipped_iter_sum[k]+skipped_iter[k] for k in range(2)]

            self.timers('train-time', log_level=0).stop()
            self.timers('end-to-end').stop()

            loss_sum = [a/total_step for a in loss_sum]
            average_reward /= total_step
            
            self.training_log(iteration, loss_sum, average_reward)
            iteration += 1
            self.args.iteration = iteration

            if self.args.empty_unused_memory_level >= 1:
                del out, loss, skipped_iter, grad_norm, loss_sum, average_reward
                torch.cuda.empty_cache()
        
        ## 保存模型
        print_rank_last("Saving Actor Model")
        save_checkpoint(iteration=iteration, model=self.actor_model, optimizer=None, opt_param_scheduler=None, model_prefix="actor")
        print_rank_last("Saving Critic Model")
        save_checkpoint(iteration=iteration, model=self.critic_model, optimizer=None, opt_param_scheduler=None, model_prefix="critic")
        
        
        return iteration

    def training_log(self, iteration, loss, average_reward):

        generate_time = self.timers('generate_sequence').elapsed()
        end2end_time = self.timers('end-to-end').elapsed()
        train_time = self.timers('train-time').elapsed()
        actor_train_time = self.timers('actor-train').elapsed()
        critic_train_time = self.timers('critic-train').elapsed()

        seq_length = self.max_seq_len
        batch_size = self.args.global_batch_size
        samples_per_second = batch_size / end2end_time
        vocab_size = self.args.padded_vocab_size

        def calculate_tflops(num_layers, hidden_size, time_):
            checkpoint_activations_factor = 3
            if hasattr(self.args, 'checkpoint_activations') and self.args.checkpoint_activations:
                checkpoint_activations_factor = 4
            if hasattr(self.args, 'recompute_granularity') and self.args.recompute_granularity == 'selective':
                checkpoint_activations_factor = 4
            flops_per_iteration = (24 * checkpoint_activations_factor * batch_size * seq_length * num_layers * (hidden_size**2)) * (
                                                1. + (seq_length / (6. * hidden_size)) + (vocab_size / (16. * num_layers * hidden_size)))
            tflops = flops_per_iteration / (time_ * (self.args.world_size / 2) * (10**12))
            
            return tflops
        
        actor_train_tflops = calculate_tflops(self.actor_config.num_layers, self.actor_config.hidden_size, actor_train_time)
        critic_train_tflops = calculate_tflops(self.critic_config.num_layers, self.critic_config.hidden_size, critic_train_time)
        actor_train_tps_device = batch_size * seq_length * 2 / self.args.world_size / actor_train_time
        critic_train_tps_device = batch_size * seq_length * 2 / self.args.world_size / critic_train_time
        
        actor_gen_flops = ( 24 * batch_size * seq_length * self.actor_config.num_layers *
            (self.actor_config.hidden_size**2)) * (
                1.0 + (seq_length / (6.0 * self.actor_config.hidden_size)) +
                (vocab_size / (16.0 * self.actor_config.num_layers * 
                self.actor_config.hidden_size))) / (generate_time * self.args.world_size * (10**12))
        
        gen_tokens_per_secend = self.args.decoder_seq_length / generate_time


        print_rank_last(f"Iteration: {iteration}, Actor model train loss: {loss[0]:.6f}, Critic model train loss: {loss[1]:.6f}")
        print_rank_last(f"End-to-End => Latency: {end2end_time:.2f}s, Samples/sec: {samples_per_second:.4f}, Time/seq {end2end_time/batch_size:.2f}s, Batch Size: {batch_size}, Total Seq. Length: {seq_length}")
        print_rank_last(f"Generation => Latency: {generate_time:.2f}s, Generate tokens/s: {gen_tokens_per_secend:.2f} , TFLOPs: {actor_gen_flops:.2f}, Answer Seq. Length: {self.args.decoder_seq_length}")
        print_rank_last(f"Training   => Latency: {train_time:.2f}s, Actor TFLOPs: {actor_train_tflops:.2f}, Critic TFLOPs: {critic_train_tflops:.2f}, Actor tokens/s/device: {actor_train_tps_device:.2f}, Critic tokens/s/device: {critic_train_tps_device:.2f}")
        print_rank_last(f"Average reward score: {average_reward}")
        print_rank_last(f"------------------------------------------------------------------------------------------------------------------------------------")


def get_model(model_provider_func, model_type=ModelType.encoder_or_decoder, wrap_with_ddp=True, rlhf_training=False):
    """Build the model."""
    if rlhf_training:
        args = get_rlhf_args()
    else:
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
                post_process=post_process,
                rlhf_training=rlhf_training
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
                post_process=post_process,
                rlhf_training=rlhf_training,
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


    # GPU allocation.
    for model_module in model:
        model_module.cuda(torch.cuda.current_device())

    # Fp16 conversion.
    if args.fp16 or args.bf16:
        model = [Float16Module(model_module, args) for model_module in model]

    if wrap_with_ddp:
        config = get_model_config(model[0])
        ddp_config = DistributedDataParallelConfig(
            grad_reduce_in_fp32=args.accumulate_allreduce_grads_in_fp32,
            overlap_grad_reduce=args.overlap_grad_reduce,
            use_distributed_optimizer=args.use_distributed_optimizer,
            check_for_nan_in_grad=args.check_for_nan_in_loss_and_grad,
            bucket_size=args.ddp_bucket_size,
            average_in_collective=args.ddp_average_in_collective)
        model = [DDP(config,
                     ddp_config,
                     model_chunk,
                     # Turn off bucketing for model_chunk 2 onwards, since communication for these
                     # model chunks is overlapped with compute anyway.
                     disable_bucketing=(model_chunk_idx > 0))
                 for (model_chunk_idx, model_chunk) in enumerate(model)]

        # Broadcast params from data parallel src rank to other data parallel ranks.
        if args.data_parallel_random_init:
            for model_module in model:
                model_module.broadcast_params()

    return model



def save_checkpoint_and_time(iteration, model, optimizer, opt_param_scheduler):
    timers = get_timers()
    # Extra barrier is added to make sure
    # all ranks report the max time.
    timers('save-checkpoint', log_level=0).start(barrier=True)
    save_checkpoint(iteration, model, optimizer, opt_param_scheduler)
    timers('save-checkpoint').stop(barrier=True)
    timers.log(['save-checkpoint'])



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


def get_batch(data_iterator):
    """Generate a batch."""

    # TODO: this is pretty hacky, find a better way
    if (not mpu.is_pipeline_first_stage()) and (not mpu.is_pipeline_last_stage()):
        return None, None, None, None, None

    args = get_args()
    tokenizer = get_tokenizer()

    # Items and their type.
    keys = ['tokens']
    datatype = torch.int64

    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    data_tokens = tensor_parallel.broadcast_data(['tokens'], data, torch.int64)
    data_labels = tensor_parallel.broadcast_data(['labels'], data, torch.int64)
    data_loss_mask = tensor_parallel.broadcast_data(['loss_mask'], data, torch.float32)
    data_attention_mask = tensor_parallel.broadcast_data(['attention_mask'], data, torch.bool) if args.create_attention_mask_in_dataloader else None
    data_position_ids = tensor_parallel.broadcast_data(['position_ids'], data, torch.int64)

    # Unpack.
    tokens = data_tokens['tokens'].contiguous()
    labels = data_labels['labels'].contiguous()
    loss_mask = data_loss_mask['loss_mask'].contiguous()
    attention_mask = data_attention_mask['attention_mask'].contiguous() if data_attention_mask is not None else None
    position_ids = data_position_ids['position_ids'].contiguous()

    # Get the masks and postition ids.
    # attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
    #     tokens,
    #     tokenizer.eod,
    #     args.reset_position_ids,
    #     args.reset_attention_mask,
    #     args.eod_mask_loss)

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


# def build_train_valid_test_datasets(build_train_valid_test_datasets_provider):
#     """Build pretraining datasets."""

#     args = get_args()

#     # Number of train/valid/test samples.
#     if args.train_samples:
#         train_samples = args.train_samples
#     else:
#         train_samples = args.train_iters * args.global_batch_size
#     eval_iters = (args.train_iters // args.eval_interval + 1) * \
#                  args.eval_iters
#     test_iters = args.eval_iters
#     train_val_test_num_samples = [train_samples,
#                                   eval_iters * args.global_batch_size,
#                                   test_iters * args.global_batch_size]
#     print_rank_0(' > datasets target sizes (minimum size):')
#     print_rank_0('    train:      {}'.format(train_val_test_num_samples[0]))
#     print_rank_0('    validation: {}'.format(train_val_test_num_samples[1]))
#     print_rank_0('    test:       {}'.format(train_val_test_num_samples[2]))

#     # Build the datasets.
#     return build_train_valid_test_datasets_provider(train_val_test_num_samples)


# def build_train_valid_test_data_loaders(
#         build_train_valid_test_datasets_provider):
#     """Build pretraining data loaders."""

#     args = get_args()

#     (train_dataloader, valid_dataloader, test_dataloader) = (None, None, None)

#     print_rank_0('> building train, validation, and test datasets ...')

#     # Backward compatibility, assume fixed batch size.
#     if args.iteration > 0 and args.consumed_train_samples == 0:
#         assert args.train_samples is None, \
#             'only backward compatiblity support for iteration-based training'
#         args.consumed_train_samples = args.iteration * args.global_batch_size
#     if args.iteration > 0 and args.consumed_valid_samples == 0:
#         if args.train_samples is None:
#             args.consumed_valid_samples = (args.iteration // args.eval_interval) * \
#                 args.eval_iters * args.global_batch_size

#     # Rely on distributed-aware core datasets, temporary
#     is_distributed = getattr(build_train_valid_test_datasets_provider, "is_distributed", False)

#     # Construct the data pipeline
#     if is_distributed or mpu.get_tensor_model_parallel_rank() == 0:

#         # Build datasets.
#         train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
#             build_train_valid_test_datasets_provider)
#         # Build dataloders.
#         train_dataloader = build_pretraining_data_loader(
#             train_ds, args.consumed_train_samples)
#         if args.skip_train:
#             valid_dataloader = build_pretraining_data_loader(valid_ds, 0)
#         else:
#             valid_dataloader = build_pretraining_data_loader(
#                 valid_ds, args.consumed_valid_samples)
#         test_dataloader = build_pretraining_data_loader(test_ds, 0)

#         # Flags to know if we need to do training/validation/testing.
#         do_train = train_dataloader is not None and args.train_iters > 0
#         do_valid = valid_dataloader is not None and args.eval_iters > 0
#         do_test = test_dataloader is not None and args.eval_iters > 0
#         flags = torch.cuda.LongTensor(
#             [int(do_train), int(do_valid), int(do_test)])
#     else:
#         flags = torch.cuda.LongTensor([0, 0, 0])

#     torch.distributed.broadcast(flags, 0)

#     args.do_train = getattr(args, "do_train", False) or flags[0].item()
#     args.do_valid = getattr(args, "do_valid", False) or flags[1].item()
#     args.do_test = getattr(args, "do_test", False) or flags[2].item()

#     return train_dataloader, valid_dataloader, test_dataloader


# def build_train_valid_test_data_iterators(
#         build_train_valid_test_datasets_provider):
#     """Build pretraining data iterators."""

#     args = get_args()

#     # Build loaders.
#     train_dataloader, valid_dataloader, test_dataloader = \
#         build_train_valid_test_data_loaders(
#             build_train_valid_test_datasets_provider)

#     # Build iterators.
#     dl_type = args.dataloader_type
#     assert dl_type in ['single', 'cyclic']

#     if train_dataloader is not None:
#         train_data_iterator = iter(train_dataloader) if dl_type == 'single' \
#                               else iter(cyclic_iter(train_dataloader))
#     else:
#         train_data_iterator = None

#     if valid_dataloader is not None:
#         valid_data_iterator = iter(valid_dataloader) if dl_type == 'single' \
#                               else iter(cyclic_iter(valid_dataloader))
#     else:
#         valid_data_iterator = None

#     if test_dataloader is not None:
#         test_data_iterator = iter(test_dataloader) if dl_type == 'single' \
#                              else iter(cyclic_iter(test_dataloader))
#     else:
#         test_data_iterator = None

#     return train_data_iterator, valid_data_iterator, test_data_iterator
