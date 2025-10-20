import torch
from torch.cuda.amp import GradScaler
from torch.types import Device
import os
import time

import utils
from model import create_model
from schedulers import create_scheduler
from optimizers.loss_scaler import DynamicLossScaler
from utils import print_rank_0
import pickle


def process_batch(batch, device):
    """Process batch and produce inputs for the model."""
    batch = {t: batch[t].to(device) for t in batch if t != 'answer_idx'}
    return batch


class Trainer:
    def __init__(self, training_event, evaluator, training_state, config):
        self.training_event = training_event
        self.evaluator = evaluator
        self.training_state = training_state
        self.config = config

        self.optimizer = None
        self.model = None
        self.lr_scheduler = None

    def init(self):
        self.model = create_model(self.config)
        self._init_model(self.model, self.config)
        self.model = self.training_event.convert_model(self.model)
        self.model = self.training_event.model_to_fp16(
            self.model)
        self.optimizer = self.training_event.create_optimizer(self.model)
        self.model = self.training_event.model_to_ddp(self.model)
        self.lr_scheduler = create_scheduler(self.optimizer, self.config)
        if self.config.fp16 and self.optimizer is not None:
            self.optimizer._model_params_to_master_params()

    def _init_model(self, model, args):
        checkpoint_name = os.path.join(
            args.init_checkpoint, '200000/mp_rank_00_model_states.pt')
        print('global rank {} is loading pretrained model {}'.format(
            torch.distributed.get_rank(), checkpoint_name))
        # Load the checkpoint.
        sd = torch.load(checkpoint_name, map_location='cpu')
        # model = model.module

        # Model.
        def extend_embedding_weights(state_weights, model_weights):
            original_length = state_weights.shape[0]
            assert original_length <= args.max_position_embeddings + 1
            new_weights = model_weights.clone()
            new_weights[:original_length] = state_weights
            return new_weights

        if "transformer.block_position_embeddings.weight" in sd["module"]:
            position_weights = sd['module']["transformer.position_embeddings.weight"]
            if args.max_position_embeddings + 1 > position_weights.shape[0]:
                sd['module']["transformer.position_embeddings.weight"] = extend_embedding_weights(
                    position_weights, model.state_dict()["transformer.position_embeddings.weight"].data)
                print_rank_0(
                    f"Extend position embedding to {args.max_position_embeddings + 1}")
        if "transformer.block_position_embeddings.weight" in sd["module"]:
            block_position_weights = sd['module']["transformer.block_position_embeddings.weight"]
            if args.max_position_embeddings + 1 > block_position_weights.shape[0]:
                sd['module']["transformer.block_position_embeddings.weight"] = extend_embedding_weights(
                    block_position_weights,
                    model.state_dict()["transformer.block_position_embeddings.weight"].data)
                print_rank_0(
                    f"Extend block position embedding to {args.max_position_embeddings + 1}")
        missing_keys, unexpected_keys = model.load_state_dict(
            sd['module'], strict=False)
        if missing_keys or unexpected_keys:
            print_rank_0(
                f"Missing keys {missing_keys}, unexpected keys {unexpected_keys}")

    def train_one_epoch(self, dataloader):
        state = self.training_state
        training_event = self.training_event

        training_event.on_epoch_begin(state.epoch)

        step_start_time = time.time()
        epoch_start_num_sample = state.num_trained_samples

        for batch_idx, batch in enumerate(dataloader):

            state.global_steps += 1
            # 考虑到最后一个batch可能会小于设定的batch size，所以 epoch完成之后，需要更新下 num_trained_samples
            state.num_trained_samples = state.global_steps * \
                utils.global_batch_size(self.config)

            self.train_one_step(batch)

            other_state = dict()
            if state.global_steps % self.config.gradient_accumulation_steps == 0:
                step_end_time = time.time()
                step_total_time = step_end_time - step_start_time
                step_start_time = step_end_time
                sequences_per_second = (utils.global_batch_size(
                    self.config) * self.config.gradient_accumulation_steps) / step_total_time
                other_state["seq/s"] = sequences_per_second

                loss_scale = self.optimizer.loss_scaler.loss_scale
                other_state['loss_scale'] = loss_scale

            eval_result = None
            if state.global_steps % 100 == 0:  # TODO: For testing
                eval_start_time = time.time()
                other_state['eval_score'] = self.evaluator.evaluate(self)
                eval_epl_time = time.time() - eval_start_time
                other_state['eval_cost_time'] = eval_epl_time
                print_rank_0(f"step:{state.global_steps} eval score:{other_state['eval_score']} time:{other_state['eval_cost_time']}")
            # TODO
            # end_training = self.detect_training_status(state)
            end_training = False

            step_info = state.to_dict(**other_state)
            print_rank_0(step_info)
            #self.training_event.on_step_end(state.global_steps, result=step_info)

            if eval_result is not None:
                self.training_event.on_evaluate(eval_result)
            
            if end_training:
                break

        epoch_start_num_sample += len(dataloader.dataset)
        state.num_trained_samples = epoch_start_num_sample
    
    # def detect_training_status(self, state: TrainingState):
    #     if state.eval_mlm_accuracy >= config.target_mlm_accuracy:
    #         state.converged_success()

    #     if state.global_steps > config.max_steps or state.num_trained_samples > config.max_samples_termination:
    #         state.end_training = True

    #     return state.end_training

    def train_one_step(self, batch):
        data = process_batch(batch, self.config.device)
        state = self.training_state

        # self.training_event.on_step_begin(state.global_steps)
        self.model.train()

        lm_loss, _ = self.forward(data)
        lm_loss /= self.config.gradient_accumulation_steps
        reduced_loss = lm_loss.detach().clone().view(1)
        torch.distributed.all_reduce(reduced_loss.data)
        reduced_loss.data = reduced_loss.data / (utils.get_world_size())

        state.average_loss = reduced_loss
        self.training_event.on_backward(
            state.global_steps, lm_loss, reduced_loss, self.optimizer, self.lr_scheduler)

        # self.training_event.on_step_end(state.global_steps)

    def forward(self, batch):
        data = batch
        tokens, labels, position_ids, attention_mask = data[
            'text'], data['label'], data['position'], data['mask']
        target_ids, logit_mask = data['target'], data['logit_mask']

        result = self.model(tokens, position_ids, attention_mask,
                            target_ids, logit_mask)
        logits, *mems = result

        loss_mask = data["loss_mask"]
        logits = logits * loss_mask - 10000.0 * (1.0 - loss_mask)

        loss_func = torch.nn.CrossEntropyLoss()
        loss = loss_func(logits.contiguous().float(), labels)

        return loss, mems

    def forward_step(self, batch, model, mems):
        data = batch
        tokens, labels, position_ids = data['text'], data['label'], data['position']
        attention_mask = data['mask']
        target_ids, logit_mask = data['target'], data['logit_mask']
        result = model(tokens, position_ids, attention_mask,
                       target_ids, logit_mask)
        logits, *mems = result

        loss_mask = data["loss_mask"]
        logits = logits * loss_mask - 10000.0 * (1.0 - loss_mask)

        loss_func = torch.nn.CrossEntropyLoss()
        loss = loss_func(logits.contiguous().float(), labels)

        return loss, mems


def train_step(data_iterator, model, optimizer, lr_scheduler, args, forward_step_func, mems=None,
               single_step=False):
    """Single training step."""
    lm_loss_total, count = 0.0, 0
    mems = [] if mems is None else mems
    args.deepspeed = False
    if not args.deepspeed:
        optimizer.zero_grad()
    while True:
        skipped_iter, complete = 0, False
        # Forward model for one step.
        lm_loss, mems = forward_step_func(data_iterator, model, mems)
        # print_rank_0("Forward step")
        if not args.deepspeed:
            lm_loss /= args.gradient_accumulation_steps

        reduced_loss = lm_loss.detach().clone().view(1)
        torch.distributed.all_reduce(reduced_loss.data)
        args.model_parallel_size = 1
        reduced_loss.data = reduced_loss.data / \
            (args.world_size / args.model_parallel_size)

        if not DynamicLossScaler._has_inf_or_nan(reduced_loss):
            lm_loss_total += reduced_loss
            count += 1

            # Calculate gradients, reduce across processes, and clip.
            backward_step(optimizer, model, lm_loss, args)
            # print_rank_0("Backward step")
            # Update parameters.
            if args.deepspeed:
                if model.is_gradient_accumulation_boundary():
                    model.step()
                    complete = True
                    if not (args.fp16 and optimizer.overflow):
                        lr_scheduler.step()
                    else:
                        skipped_iter = 1
                else:
                    model.step()
            else:
                if count == args.gradient_accumulation_steps:
                    optimizer.step()
                    complete = True
                    # Update learning rate.
                    if not (args.fp16 and optimizer.overflow):
                        lr_scheduler.step()
                    else:
                        skipped_iter = 1
            # print_rank_0("Optimizer step")
            if complete:
                break
        else:
            print_rank_0("Found NaN loss, skip backward")
            del lm_loss, reduced_loss
            mems = []
        if single_step:
            break
    if args.deepspeed:
        lm_loss_total = lm_loss_total / count
    return lm_loss_total, skipped_iter, mems


def backward_step(optimizer, model, lm_loss, args):
    """Backward step."""

    # Total loss.
    loss = lm_loss

    # Backward pass.
    if args.deepspeed:
        model.backward(loss)
    else:
        # optimizer.zero_grad()
        if args.fp16:
            optimizer.backward(loss, update_master_grads=False)
        else:
            loss.backward()
    args.DDP_impl = 'torch'
    if not (args.deepspeed or args.DDP_impl == 'torch'):
        model.allreduce_params(reduce_after=False, fp32_allreduce=False)

    # Update master gradients.
    if not args.deepspeed:
        if args.fp16:
            optimizer.update_master_grads()

        # Clipping gradients helps prevent the exploding gradient.
        if args.clip_grad > 0:
            if not args.fp16:
                mpu.clip_grad_norm(model.parameters(), args.clip_grad)
            else:
                optimizer.clip_master_grads(args.clip_grad)

    return lm_loss
