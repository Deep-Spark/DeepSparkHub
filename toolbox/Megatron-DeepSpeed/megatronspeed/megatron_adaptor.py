import sys
import argparse
from functools import wraps
import torch
from torch.distributed import all_gather_into_tensor, reduce_scatter_tensor
from megatronspeed.training.arguments import process_args

_ARGS = None

IS_ADAPTED = False

def add_args(args, key, value):
    if key is not None:
        key = key[2:].replace('-', '_')
        if value is None:
            value = True
        elif len(value) == 1:
            value = value[0]
        setattr(args, key, value)


def parser_unknown_args(args, unknown):
    i = 0
    key = value = None
    while i < len(unknown):
        if unknown[i].startswith("--"):
            add_args(args, key, value)
            key = unknown[i]
            value = None
        else:
            if value is None:
                value = [unknown[i]]
            else:
                value.append(unknown[i])
        i += 1
    add_args(args, key, value)


def version_wrapper(fn):
    @wraps(fn)
    def wrapper(name, *args, **kwargs):
        if name == 'transformer-engine':
            return '0.0'
        res = fn(name, *args, **kwargs)
        return res

    return wrapper


def te_adaptation(aspm):
    # Need replace modules before import megatron
    # aspm.register_patch('importlib.metadata.version', version_wrapper)
    pass


def apex_adaptation(aspm):
    pass


def torch_adaptation(aspm):
    aspm.register_patch('torch.distributed._all_gather_base', all_gather_into_tensor)
    aspm.register_patch('torch.distributed._reduce_scatter_base', reduce_scatter_tensor)


def mcore_models_adaptation(aspm):
    from .core.utils import get_model_config

    aspm.register_patch('megatron.core.utils.get_model_config', get_model_config)


def preparation_adaption(aspm):
    from .training.global_vars import get_rlhf_args, set_rlhf_args

    aspm.register_patch('megatron.training.global_vars.get_rlhf_args', get_rlhf_args)
    aspm.register_patch('megatron.training.global_vars.set_rlhf_args', set_rlhf_args)
    aspm.register_patch('megatron.training.get_rlhf_args', get_rlhf_args)
    aspm.register_patch('megatron.training.set_rlhf_args', set_rlhf_args)


def mcore_tensor_parallel_adaptation(aspm):
    from .core.tensor_parallel.random import init_checkpointed_activations_memory_buffer, reset_checkpointed_activations_memory_buffer, \
        get_cuda_rng_tracker, model_parallel_cuda_manual_seed, model_parallel_reconfigure_tp_seed, checkpoint, \
        checkpoint_function_forward, checkpoint_function_backward
    from .core.tensor_parallel.layers import linear_with_grad_accumulation_and_async_allreduce_forward, linear_with_grad_accumulation_and_async_allreduce_backward, \
        linear_with_grad_accumulation_and_async_allreduce, SequenceParallelPositionEmbedding, column_parallel_linear_init, column_parallel_linear_forward, \
        row_parallel_linear_init, row_parallel_linear_forward
    from .core.tensor_parallel.data import _build_key_size_numel_dictionaries, broadcast_data

    aspm.register_patch('megatron.core.tensor_parallel.random.init_checkpointed_activations_memory_buffer', init_checkpointed_activations_memory_buffer)
    aspm.register_patch('megatron.core.tensor_parallel.random.reset_checkpointed_activations_memory_buffer', reset_checkpointed_activations_memory_buffer)
    aspm.register_patch('megatron.core.tensor_parallel.random.get_cuda_rng_tracker', get_cuda_rng_tracker)
    aspm.register_patch('megatron.core.tensor_parallel.random.model_parallel_cuda_manual_seed', model_parallel_cuda_manual_seed)
    aspm.register_patch('megatron.core.tensor_parallel.random.model_parallel_reconfigure_tp_seed', model_parallel_reconfigure_tp_seed)
    aspm.register_patch('megatron.core.tensor_parallel.random.CheckpointFunction.forward', checkpoint_function_forward)
    aspm.register_patch('megatron.core.tensor_parallel.random.CheckpointFunction.backward', checkpoint_function_backward)
    aspm.register_patch('megatron.core.tensor_parallel.random.checkpoint', checkpoint)
    aspm.register_patch('megatron.core.tensor_parallel.layers.LinearWithGradAccumulationAndAsyncCommunication.forward',
                        linear_with_grad_accumulation_and_async_allreduce_forward)
    aspm.register_patch('megatron.core.tensor_parallel.layers.LinearWithGradAccumulationAndAsyncCommunication.backward',
                        linear_with_grad_accumulation_and_async_allreduce_backward)
    aspm.register_patch('megatron.core.tensor_parallel.layers.linear_with_grad_accumulation_and_async_allreduce', linear_with_grad_accumulation_and_async_allreduce)
    aspm.register_patch('megatron.core.tensor_parallel.layers.SequenceParallelPositionEmbedding', SequenceParallelPositionEmbedding)
    aspm.register_patch('megatron.core.tensor_parallel.layers.ColumnParallelLinear.__init__', column_parallel_linear_init)
    aspm.register_patch('megatron.core.tensor_parallel.layers.ColumnParallelLinear.forward', column_parallel_linear_forward)
    aspm.register_patch('megatron.core.tensor_parallel.layers.RowParallelLinear.__init__', row_parallel_linear_init)
    aspm.register_patch('megatron.core.tensor_parallel.layers.RowParallelLinear.forward', row_parallel_linear_forward)
    aspm.register_patch('megatron.core.tensor_parallel.data._build_key_size_numel_dictionaries', _build_key_size_numel_dictionaries)
    aspm.register_patch('megatron.core.tensor_parallel.data.broadcast_data', broadcast_data)


def mcore_pipeline_parallel_adaptation(aspm):
    from .core.pipeline_parallel.schedules import backward_step, forward_backward_no_pipelining, forward_backward_pipelining_without_interleaving

    aspm.register_patch('megatron.core.pipeline_parallel.schedules.backward_step', backward_step)
    aspm.register_patch('megatron.core.pipeline_parallel.schedules.forward_backward_no_pipelining', forward_backward_no_pipelining)
    aspm.register_patch('megatron.core.pipeline_parallel.schedules.forward_backward_pipelining_without_interleaving',
                        forward_backward_pipelining_without_interleaving)


def mcore_transformer_adaptation(aspm):
    from .core.transformer.utils import get_linear_layer

    aspm.register_patch('megatron.core.transformer.utils.get_linear_layer', get_linear_layer)

def legacy_model_transformer(aspm):
    from .legacy.model.module import megatron_module_universal_checkpoint_info
    from .legacy.model.utils import gather_and_init, attention_mask_func, get_linear_layer
    from .legacy.model.transformer import parallel_mlp_init, parallel_mlp_forward_wrapper, core_attention_init, flash_selfattention_forward_wrapper, parallel_attention_init, \
        parallel_attention_forward_wrapper, parallel_transformer_layer_init, parallel_transformer_layer_forward_wrapper, parallel_transformer_init, \
        parallel_transformer__checkpointed_forward_wrapper, parallel_transformer_forward_wrapper
    from .legacy.model.realm_model import IREncoderBertModel
    from .legacy.model.multiple_choice import MultipleChoice
    from .legacy.model.language_model import parallel_lm_logits, get_language_model, pooler_init, embedding_init, embedding_forward, transformer_language_model_init, \
        transformer_language_model_forward_wrapper, transformer_language_model_state_dict_for_save_checkpoint, transformer_language_model_load_state_dict
    from .legacy.model.gpt_model import post_language_model_processing, gpt_model_init, gpt_model_forward_wrapper, gpt_model_state_dict_for_save_checkpoint, \
        gpt_model_load_state_dict, gpt_model_universal_checkpoint_info, GPTModelPipe
    from .legacy.model.classification import Classification
    from .legacy.model.bert_model import BertModel
    from .legacy.model.t5_model import T5Model
    from .legacy.model.biencoder_model import PretrainedBertModel

    aspm.register_patch('megatron.legacy.model.module.MegatronModule.universal_checkpoint_info', megatron_module_universal_checkpoint_info, create_dummy=True)
    aspm.register_patch('megatron.legacy.model.utils.gather_and_init', gather_and_init)
    aspm.register_patch('megatron.legacy.model.utils.attention_mask_func', attention_mask_func)
    aspm.register_patch('megatron.legacy.model.utils.get_linear_layer', get_linear_layer)
    aspm.register_patch('megatron.legacy.model.transformer.ParallelMLP.__init__', parallel_mlp_init)
    aspm.register_patch('megatron.legacy.model.transformer.ParallelMLP.forward', parallel_mlp_forward_wrapper)
    aspm.register_patch('megatron.legacy.model.transformer.CoreAttention.__init__', core_attention_init)
    aspm.register_patch('megatron.legacy.model.transformer.FlashSelfAttention.forward', flash_selfattention_forward_wrapper)
    aspm.register_patch('megatron.legacy.model.transformer.ParallelAttention.__init__', parallel_attention_init)
    aspm.register_patch('megatron.legacy.model.transformer.ParallelAttention.forward', parallel_attention_forward_wrapper)
    aspm.register_patch('megatron.legacy.model.transformer.ParallelTransformerLayer.__init__', parallel_transformer_layer_init)
    aspm.register_patch('megatron.legacy.model.transformer.ParallelTransformerLayer.forward', parallel_transformer_layer_forward_wrapper)
    aspm.register_patch('megatron.legacy.model.transformer.ParallelTransformer.__init__', parallel_transformer_init)
    aspm.register_patch('megatron.legacy.model.transformer.ParallelTransformer._checkpointed_forward', parallel_transformer__checkpointed_forward_wrapper)
    aspm.register_patch('megatron.legacy.model.transformer.ParallelTransformer.forward', parallel_transformer_forward_wrapper)
    aspm.register_patch('megatron.legacy.model.language_model.parallel_lm_logits', parallel_lm_logits)
    aspm.register_patch('megatron.legacy.model.language_model.get_language_model', get_language_model)
    aspm.register_patch('megatron.legacy.model.language_model.Pooler.__init__', pooler_init)
    aspm.register_patch('megatron.legacy.model.language_model.Embedding.__init__', embedding_init)
    aspm.register_patch('megatron.legacy.model.language_model.Embedding.forward', embedding_forward)
    aspm.register_patch('megatron.legacy.model.language_model.TransformerLanguageModel.__init__',
                        transformer_language_model_init)
    aspm.register_patch('megatron.legacy.model.language_model.TransformerLanguageModel.forward',
                        transformer_language_model_forward_wrapper)
    aspm.register_patch('megatron.legacy.model.language_model.TransformerLanguageModel.state_dict_for_save_checkpoint',
                        transformer_language_model_state_dict_for_save_checkpoint)
    aspm.register_patch('megatron.legacy.model.language_model.TransformerLanguageModel.load_state_dict',
                        transformer_language_model_load_state_dict)
    aspm.register_patch('megatron.legacy.model.gpt_model.post_language_model_processing', post_language_model_processing)
    aspm.register_patch('megatron.legacy.model.gpt_model.GPTModel.__init__', gpt_model_init)
    aspm.register_patch('megatron.legacy.model.gpt_model.GPTModel.forward', gpt_model_forward_wrapper)
    aspm.register_patch('megatron.legacy.model.gpt_model.GPTModel.state_dict_for_save_checkpoint', gpt_model_state_dict_for_save_checkpoint)
    aspm.register_patch('megatron.legacy.model.gpt_model.GPTModel.load_state_dict', gpt_model_load_state_dict)
    aspm.register_patch('megatron.legacy.model.gpt_model.GPTModel.universal_checkpoint_info', gpt_model_universal_checkpoint_info)
    aspm.register_patch('megatron.legacy.model.GPTModelPipe', GPTModelPipe)
    aspm.register_patch('megatron.legacy.model.realm_model.IREncoderBertModel', IREncoderBertModel)
    aspm.register_patch('megatron.legacy.model.multiple_choice.MultipleChoice', MultipleChoice)
    aspm.register_patch('megatron.legacy.model.classification.Classification', Classification)
    aspm.register_patch('megatron.legacy.model.bert_model.BertModel', BertModel)
    aspm.register_patch('megatron.legacy.model.t5_model.T5Model', T5Model)
    aspm.register_patch('megatron.legacy.model.biencoder_model.PretrainedBertModel', PretrainedBertModel)


def legacy_data_base(aspm):
    from .legacy.data.blendable_dataset import BlendableDataset
    from .legacy.data.indexed_dataset import make_dataset

    aspm.register_patch('megatron.legacy.data.blendable_dataset.BlendableDataset', BlendableDataset, create_dummy=True)
    aspm.register_patch('megatron.legacy.data.indexed_dataset.make_dataset', make_dataset, create_dummy=True)


def legacy_data_adaption(aspm):
    from .legacy.data.gpt_dataset import GPTDataset, build_train_valid_test_datasets

    aspm.register_patch('megatron.legacy.data.gpt_dataset.GPTDataset', GPTDataset, create_dummy=True)
    aspm.register_patch('megatron.legacy.data.gpt_dataset.build_train_valid_test_datasets', build_train_valid_test_datasets, create_dummy=True)




def mcore_optimizer_adapation(aspm):
    from .core.optimizer import get_param_groups, _get_param_groups_mod, get_megatron_optimizer_wrapper

    aspm.register_patch('megatron.core.optimizer.get_param_groups', get_param_groups)
    aspm.register_patch('megatron.core.optimizer._get_param_groups', _get_param_groups_mod)
    aspm.register_patch('megatron.core.optimizer.get_megatron_optimizer', get_megatron_optimizer_wrapper)


def megatron_training_adaptation(aspm):
    from .training.arguments import parse_args_wrapper, validate_args
    from .training.training import pretrain, get_model, setup_model_and_optimizer, train_step, training_log, train, evaluate, \
        evaluate_and_print_results, build_train_valid_test_data_loaders
    from .training.initialize import initialize_megatron, _compile_dependencies, _initialize_distributed, _warmup_jit_function
    from .training.checkpointing import check_checkpoint_args, save_checkpoint, generate_state_dict, load_checkpoint
    from .training.utils import get_ltor_masks_and_position_ids, update_rotary_pos_emb
    from .training.tokenizer import build_tokenizer

    aspm.register_patch('megatron.training.utils.get_ltor_masks_and_position_ids', get_ltor_masks_and_position_ids)
    aspm.register_patch('megatron.training.utils.update_rotary_pos_emb', update_rotary_pos_emb)
    aspm.register_patch('megatron.training.arguments.parse_args', parse_args_wrapper)
    aspm.register_patch('megatron.training.arguments.validate_args', validate_args)
    aspm.register_patch('megatron.training.yaml_arguments.validate_yaml', validate_args)
    aspm.register_patch('megatron.training.training.pretrain', pretrain)
    aspm.register_patch('megatron.training.training.train', train)
    aspm.register_patch('megatron.training.training.get_model', get_model)
    aspm.register_patch('megatron.training.training.setup_model_and_optimizer', setup_model_and_optimizer)
    aspm.register_patch('megatron.training.training.train_step', train_step)
    aspm.register_patch('megatron.training.training.training_log', training_log)
    aspm.register_patch('megatron.training.training.evaluate', evaluate)
    aspm.register_patch('megatron.training.training.evaluate_and_print_results', evaluate_and_print_results)
    aspm.register_patch('megatron.training.training.build_train_valid_test_data_loaders', build_train_valid_test_data_loaders)
    aspm.register_patch('megatron.training.initialize.initialize_megatron', initialize_megatron)
    aspm.register_patch('megatron.training.initialize._compile_dependencies', _compile_dependencies)
    aspm.register_patch('megatron.training.initialize._initialize_distributed', _initialize_distributed)
    aspm.register_patch('megatron.training.initialize._warmup_jit_function', _warmup_jit_function)
    aspm.register_patch('megatron.training.checkpointing.check_checkpoint_args', check_checkpoint_args)
    aspm.register_patch('megatron.training.checkpointing.save_checkpoint', save_checkpoint)
    aspm.register_patch('megatron.training.checkpointing.generate_state_dict', generate_state_dict)
    aspm.register_patch('megatron.training.checkpointing.load_checkpoint', load_checkpoint)
    aspm.register_patch('megatron.training.tokenizer.tokenizer.build_tokenizer', build_tokenizer)


def mcore_parallel_state_adaptation(aspm):
    from .core.parallel_state import initialize_model_parallel_wrapper, destroy_model_parallel_wrapper
    from .core.parallel_state import sequence_parallel_is_initialized, sequence_data_parallel_is_initialized, \
        get_sequence_parallel_group, get_sequence_data_parallel_group, set_sequence_parallel_world_size, set_sequence_data_parallel_world_size, \
        get_model_parallel_world_size, get_sequence_parallel_world_size, get_sequence_data_parallel_world_size, get_model_parallel_rank, \
        set_sequence_parallel_rank, set_sequence_data_parallel_rank, get_sequence_parallel_rank, get_sequence_data_parallel_rank, \
        get_sequence_parallel_src_rank

    aspm.register_patch('megatron.core.parallel_state.initialize_model_parallel', initialize_model_parallel_wrapper)
    aspm.register_patch('megatron.core.parallel_state.destroy_model_parallel', destroy_model_parallel_wrapper)
    aspm.register_patch('megatron.core.parallel_state.sequence_parallel_is_initialized', sequence_parallel_is_initialized)
    aspm.register_patch('megatron.core.parallel_state.sequence_data_parallel_is_initialized', sequence_data_parallel_is_initialized)
    aspm.register_patch('megatron.core.parallel_state.get_sequence_parallel_group', get_sequence_parallel_group)
    aspm.register_patch('megatron.core.parallel_state.get_sequence_data_parallel_group', get_sequence_data_parallel_group)
    aspm.register_patch('megatron.core.parallel_state.set_sequence_parallel_world_size', set_sequence_parallel_world_size)
    aspm.register_patch('megatron.core.parallel_state.set_sequence_data_parallel_world_size', set_sequence_data_parallel_world_size)
    aspm.register_patch('megatron.core.parallel_state.get_model_parallel_world_size', get_model_parallel_world_size)
    aspm.register_patch('megatron.core.parallel_state.get_sequence_parallel_world_size', get_sequence_parallel_world_size)
    aspm.register_patch('megatron.core.parallel_state.get_sequence_data_parallel_world_size', get_sequence_data_parallel_world_size)
    aspm.register_patch('megatron.core.parallel_state.get_model_parallel_rank', get_model_parallel_rank)
    aspm.register_patch('megatron.core.parallel_state.set_sequence_parallel_rank', set_sequence_parallel_rank)
    aspm.register_patch('megatron.core.parallel_state.set_sequence_data_parallel_rank', set_sequence_data_parallel_rank)
    aspm.register_patch('megatron.core.parallel_state.get_sequence_parallel_rank', get_sequence_parallel_rank)
    aspm.register_patch('megatron.core.parallel_state.get_sequence_data_parallel_rank', get_sequence_data_parallel_rank)
    aspm.register_patch('megatron.core.parallel_state.get_sequence_parallel_src_rank', get_sequence_parallel_src_rank)


def adaptation_l0(aspm):
    """
    The minimum patch set for megatron to adapt to NPU
    """
    # transformer_engine
    te_adaptation(aspm)
    apex_adaptation(aspm)
    torch_adaptation(aspm)
    legacy_data_base(aspm)
    preparation_adaption(aspm)
    # Need replace transformer_engine modules before import megatron
    aspm.apply_patches()

    mcore_models_adaptation(aspm)
    mcore_tensor_parallel_adaptation(aspm)
    mcore_pipeline_parallel_adaptation(aspm)
    mcore_transformer_adaptation(aspm)
    legacy_model_transformer(aspm)
    legacy_data_adaption(aspm)
    mcore_optimizer_adapation(aspm)
    megatron_training_adaptation(aspm)
    mcore_parallel_state_adaptation(aspm)


def get_megatronspeed_args():
    global _ARGS
    if _ARGS is None:
        parser = argparse.ArgumentParser(description='Megatron-Deepspeed Arguments', allow_abbrev=False)
        _ARGS, unknown = process_args(parser).parse_known_args()
        parser_unknown_args(_ARGS, unknown)
    return _ARGS


def exe_adaptation():
    global IS_ADAPTED
    if IS_ADAPTED:
        return

    megatronspeed_args = get_megatronspeed_args()
    from .patch_utils import MegatronPatchesManager as aspm

    adaptation_l0(aspm)

    aspm.apply_patches()

    IS_ADAPTED = True


exe_adaptation()
