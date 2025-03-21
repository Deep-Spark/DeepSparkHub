# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

"""GPT-2 model."""

import torch

from megatron_ds import get_args, get_rlhf_args
from megatron_ds.core import mpu, tensor_parallel, sequence_parallel
from .module import MegatronModule, fp32_to_float16, float16_to_fp32

from .enums import AttnMaskType
from .language_model import parallel_lm_logits
from .language_model import get_language_model
from .utils import init_method_normal
from .utils import scaled_init_method_normal

from megatron_ds.model import LayerNorm,RMSNorm
from .language_model import EmbeddingPipe
from .transformer import ParallelTransformerLayerPipe, LMHeadPipe
from deepspeed.pipe import PipelineModule, LayerSpec, TiedLayerSpec
import ixformer.train.functions as IXF
from typing import List, Sequence

try:         
    from deepspeed.checkpoint import (
        VOCABULARY_PARAMETER_PATTERNS,
        PIPELINE_REPLICATED_PARAMETER_PATTERNS,
        TP_REPLICATED_PARAMETER_PATTERNS,
        PARAMETER_WITH_ROW_PARALLELISM_PATTERNS,
        PARAMETER_WITH_2_SUB_PARAMS_CAT_DIM_0,
    )
    DS_UNIVERSAL_CHECKPOINT_INFO = True 
except ImportError:
    DS_UNIVERSAL_CHECKPOINT_INFO = False  


def vocab_range_from_per_partition_vocab_size(
        per_partition_vocab_size: int, rank, world_size: int
    ) -> Sequence[int]:
        index_f = rank * per_partition_vocab_size
        index_l = index_f + per_partition_vocab_size
        return index_f, index_l

def post_language_model_processing(lm_output, labels, logit_weights,
                                   parallel_output,
                                   fp16_lm_cross_entropy, inference_params=None):
    # Output. Format [s b h]
    output = parallel_lm_logits(
        lm_output,
        logit_weights,
        parallel_output,
        inference_params=inference_params)

    if labels is None:
        # [s b h] => [b s h]
        return output.transpose(0,1).contiguous()
    else:
        # [b s] => [s b]
        labels = labels.transpose(0,1).contiguous()
        cross_entropy = sequence_parallel.vocab_sequence_parallel_cross_entropy if mpu.get_sequence_parallel_world_size() > 1 \
            else tensor_parallel.vocab_parallel_cross_entropy
        if fp16_lm_cross_entropy:
            assert output.dtype == torch.half
            loss = cross_entropy(output, labels)
        else:
            # loss = cross_entropy(output.float(), labels)
            get_vocab_range = vocab_range_from_per_partition_vocab_size
            partition_vocab_size = output.size()[-1]
            rank = mpu.get_tensor_model_parallel_rank()
            world_size = mpu.get_tensor_model_parallel_world_size()
            group = mpu.get_tensor_model_parallel_group()
            vocab_start_index, vocab_end_index = get_vocab_range(partition_vocab_size, rank, world_size)
            loss = IXF.vocab_parallel_cross_entropy(
                output,
                labels,
                0.0,
                world_size,
                vocab_start_index,
                vocab_end_index,
                group
            )
        # [s b] => [b, s]
        loss = loss.transpose(0,1).contiguous()
        return loss


class GPTModel(MegatronModule):
    """GPT-2 Language model."""

    def __init__(self,
                 config,
                 num_tokentypes=0,
                 parallel_output=True,
                 pre_process=True,
                 post_process=True,
                 return_moe_loss=True,
                 rlhf_training=False):
        self.rlhf_training = rlhf_training
        if rlhf_training:
            args = get_rlhf_args()
        else:
            args = get_args()

        super().__init__(config=config, share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights)

        self.parallel_output = parallel_output
        self.pre_process = pre_process
        self.post_process = post_process
        self.fp16_lm_cross_entropy = args.fp16_lm_cross_entropy
        self.return_moe_loss = return_moe_loss
        self.untie_embeddings_and_output_weights = args.untie_embeddings_and_output_weights

        self.language_model, self._language_model_key = get_language_model(
            config=config,
            num_tokentypes=num_tokentypes,
            add_pooler=False,
            encoder_attn_mask_type=AttnMaskType.causal,
            pre_process=self.pre_process,
            post_process=self.post_process,
            num_experts=args.num_experts,
            rlhf_training=rlhf_training)

        if not args.untie_embeddings_and_output_weights:
            self.initialize_word_embeddings()

    def set_input_tensor(self, input_tensor):
        """See megatron_ds.model.transformer.set_input_tensor()"""
        self.language_model.set_input_tensor(input_tensor)

    def forward(self, input_ids, position_ids, attention_mask,
                retriever_input_ids=None,
                retriever_position_ids=None,
                retriever_attn_mask=None,
                labels=None, tokentype_ids=None, inference_params=None,
                curriculum_seqlen=None, parallel_output=None):
        args = get_args()

        if curriculum_seqlen is not None:
            args.curriculum_seqlen = curriculum_seqlen
            if curriculum_seqlen < input_ids.size()[1]:
                # seqlen-based curriculum learning
                # input_ids, position_ids, labels have size [batch size, seqlen]
                input_ids = input_ids[:, :curriculum_seqlen].contiguous()
                position_ids = position_ids[:, :curriculum_seqlen].contiguous()
                if labels is not None:
                    labels = labels[:, :curriculum_seqlen].contiguous()

                # attention_mask has size [1, 1, seqlen, seqlen]
                attention_mask = attention_mask[:, :, :curriculum_seqlen, :curriculum_seqlen].contiguous()
        else:
            if args.curriculum_learning_legacy:
                # If got a None input, need to reset curriculum_seqlen on user side
                args.curriculum_seqlen = args.seq_length

        lm_output = self.language_model(
            input_ids,
            position_ids,
            attention_mask,
            retriever_input_ids=retriever_input_ids,
            retriever_position_ids=retriever_position_ids,
            retriever_attn_mask=retriever_attn_mask,
            inference_params=inference_params) # [s, b, h]

        if self.post_process:
            if self.rlhf_training and self.untie_embeddings_and_output_weights:
                # Run rlhf last linear layer, which mapping hidden_size to 1

                lm_output = self.language_model.output_layer(lm_output).squeeze(-1)
                lm_output = lm_output.transpose(0,1).contiguous() # [s b] => [b, s]
                if args.sequence_parallel:
                    lm_output = tensor_parallel.gather_from_tensor_model_parallel_region(lm_output)

                return lm_output
            else:
                if parallel_output is not None:
                    # Use input parallel_output during inference phase to avoid using default self.parallel_output in model init
                    # To get the complete output during inference phase, we should set parallel_output=True
                    lm_output = post_language_model_processing(
                        lm_output, labels,
                        self.language_model.output_layer.weight if self.untie_embeddings_and_output_weights else self.shared_embedding_or_output_weight(),
                        parallel_output,
                        self.fp16_lm_cross_entropy,
                        inference_params=inference_params)
                else:
                    lm_output = post_language_model_processing(
                        lm_output, labels,
                        self.language_model.output_layer.weight if self.untie_embeddings_and_output_weights else self.shared_embedding_or_output_weight(),
                        self.parallel_output,
                        self.fp16_lm_cross_entropy,
                        inference_params=inference_params)

        return lm_output

    def state_dict_for_save_checkpoint(self, prefix='', keep_vars=False):

        state_dict_ = {}
        language_model_state_dict = self.language_model.state_dict_for_save_checkpoint(
                prefix=prefix, keep_vars=keep_vars)
        # MoE states need to be handled separately by DeepSpeed engine, thus
        # moving them to the top level dictionary
        if "moe_state_dict" in language_model_state_dict:
            for key in list(language_model_state_dict["moe_state_dict"].keys()):
                state_dict_[key] = language_model_state_dict["moe_state_dict"].pop(key)
            del language_model_state_dict["moe_state_dict"]
        state_dict_[self._language_model_key] = language_model_state_dict
        # Save word_embeddings.
        if self.post_process and not self.pre_process and not self.untie_embeddings_and_output_weights:
            state_dict_[self._word_embeddings_for_head_key] \
                = self.word_embeddings.state_dict(prefix=prefix,
                                                  keep_vars=keep_vars)
        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        """Customized load."""

        # Load word_embeddings.
        if self.post_process and not self.pre_process and not self.untie_embeddings_and_output_weights:
            self.word_embeddings.load_state_dict(
                state_dict[self._word_embeddings_for_head_key], strict=strict)
        # Gather MoE states and move under language model
        moe_state_dict = {}
        for key in list(state_dict.keys()):
            if 'expert' in key and 'moe.gate.wg.weight' not in key:
                moe_state_dict[key] = state_dict.pop(key)
        if self._language_model_key in state_dict:
            state_dict = state_dict[self._language_model_key]
        if len(moe_state_dict) > 0:
            state_dict["moe_state_dict"] = moe_state_dict
        self.language_model.load_state_dict(state_dict, strict=strict)

    def _get_vocab_param_patterns(self):
        args = get_args()
        if args.untie_embeddings_and_output_weights:
            patterns = [
                r"\d+.word_embeddings.weight",
                r"\d+.lm_head.weight"
            ]
        else:
            patterns = [
                r"tied_modules.embed.word_embeddings.weight"
            ]
        return patterns

    def universal_checkpoint_info(self):
        info = dict()
        args = get_args()

        if DS_UNIVERSAL_CHECKPOINT_INFO:
            # Vocabulary parameters (embeddings) that require special handling due to padding.
            info[VOCABULARY_PARAMETER_PATTERNS] = self._get_vocab_param_patterns()
            
            if args.tensor_model_parallel_size > 1:
                # Parameter slices that should be averaged not concatenated.
                info[TP_REPLICATED_PARAMETER_PATTERNS] = self._get_tp_replicated_param_patterns()

                # Parameter that are sliced on the row dimension
                info[PARAMETER_WITH_ROW_PARALLELISM_PATTERNS] = self._get_row_parallel_param_patterns()

        return info
    
def CrossEntropy(output, labels):
    labels, loss_mask = labels[0], labels[1]

    args = get_args()

    # [b s] => [s b]
    labels = labels.transpose(0, 1).contiguous()
    losses = tensor_parallel.vocab_parallel_cross_entropy(output.contiguous().float(), labels)
    # [s b] => [b, s]
    losses = losses.transpose(0, 1).contiguous()
    loss_mask = loss_mask.view(-1)
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()
    return loss


class GPTModelPipe(PipelineModule,MegatronModule):
    """GPT-2 Language model."""

    def __init__(self,
                 config,
                 num_tokentypes=0,
                 parallel_output=True,
                 partition_method='uniform',
                 custom_partition=None):
        args = get_args()
        self.parallel_output = parallel_output

        if config.init_method is None:
            config.init_method = init_method_normal(config.init_method_std)

        if config.output_layer_init_method is None:
            config.output_layer_init_method = scaled_init_method_normal(config.init_method_std,
                                                                        config.num_layers)

        self.specs = []

        def _to_float16(inputs):
            if args.fp16:
                return fp32_to_float16(inputs, lambda v: v.half())
            elif args.bf16:
                return fp32_to_float16(inputs, lambda v: v.bfloat16())
            else:
                return inputs

        self.specs.append(_to_float16)

        # Embedding layer
        if args.untie_embeddings_and_output_weights:
            self.specs.append(LayerSpec(EmbeddingPipe,
                                        args.hidden_size,
                                        args.padded_vocab_size,
                                        args.max_position_embeddings,
                                        args.hidden_dropout,
                                        config,
                                        num_tokentypes=num_tokentypes,
                                        embedding_weights_in_fp32=args.embedding_weights_in_fp32,))
        else:
            self.specs.append(TiedLayerSpec('embed',
                                            EmbeddingPipe,
                                            args.hidden_size,
                                            args.padded_vocab_size,
                                            args.max_position_embeddings,
                                            args.hidden_dropout,
                                            config,
                                            num_tokentypes=num_tokentypes,
                                            embedding_weights_in_fp32=args.embedding_weights_in_fp32,
                                            tied_weight_attr='word_embeddings_weight'))

        for layer_idx in range(args.num_layers):
            self.specs.append(
                LayerSpec(ParallelTransformerLayerPipe,
                    config,
                    layer_number=layer_idx,
                    self_attn_mask_type=AttnMaskType.causal))

        # Final layernorm after transformer layers
        if args.normalization == 'layernorm':
            self.specs.append(LayerSpec(LayerNorm,
                          args.hidden_size,
                          eps=args.layernorm_epsilon))
        else:
            self.specs.append(LayerSpec(RMSNorm, args.hidden_size, args.layernorm_epsilon))

        def _logits_helper(embedding, lm_output):
            """A wrapper to massage inputs/outputs from pipeline. """
            return parallel_lm_logits(
                lm_output,
                embedding.word_embeddings_weight,
                self.parallel_output)
        if args.untie_embeddings_and_output_weights:
            self.specs.append(
                LayerSpec(LMHeadPipe, args.hidden_size, args.padded_vocab_size, config)
            )
        else:
            self.specs.append(
                TiedLayerSpec('embed',
                              EmbeddingPipe,
                              args.hidden_size,
                              args.padded_vocab_size,
                              args.max_position_embeddings,
                              args.hidden_dropout,
                              config,
                              num_tokentypes=num_tokentypes,
                              embedding_weights_in_fp32=args.embedding_weights_in_fp32,
                              forward_fn=_logits_helper,
                              tied_weight_attr='word_embeddings_weight')
            )

        # Convert to fp32 if needed
        if args.fp16 or args.bf16:
            self.specs.append(float16_to_fp32)

        if args.checkpoint_activations:
            interval = args.checkpoint_num_layers
        elif args.recompute_granularity == "full" and args.recompute_method == 'uniform':
            # deepspeed's pipeline doesn't support the block recompute method
            interval = args.recompute_num_layers
        else:
            interval = 0

        from deepspeed.runtime.pipe.topology import PipeModelDataParallelTopology
        topo = PipeModelDataParallelTopology(num_pp=mpu.get_pipeline_model_parallel_world_size(),
                                             num_mp=mpu.get_tensor_model_parallel_world_size(),
                                             num_dp=mpu.get_data_parallel_world_size())

        super().__init__(layers=self.specs,
                         loss_fn=CrossEntropy,
                         topology=topo,
                         activation_checkpoint_interval=interval,
                         partition_method=partition_method,
                         custom_partition=custom_partition,
                         custom_recompute_layers_per_stage=args.custom_recompute_layers_per_stage)

    @staticmethod
    def _get_vocab_param_patterns():
        args = get_args()
        if args.untie_embeddings_and_output_weights:
            patterns = [
                r"\d+.word_embeddings.weight",
                r"\d+.lm_head.weight"
            ]
        else:
            patterns = [
                r"tied_modules.embed.word_embeddings.weight"
            ]
        return patterns

    def _get_pp_replicated_param_patterns(self):
        args = get_args()
        if args.untie_embeddings_and_output_weights:
            return []
        patterns = self._get_vocab_param_patterns()
        if args.add_position_embedding:
            patterns.append(r"tied_modules.embed.position_embeddings.weight")
        return patterns

    @staticmethod
    def _get_tp_replicated_param_patterns():
        args = get_args()
        patterns = [
            r"\d+.input_layernorm.weight",
            r"\d+.post_attention_layernorm.weight",
            r"\d+.weight",
        ]
        if args.add_position_embedding:
            patterns.append(r"tied_modules.embed.position_embeddings.weight")
        if args.add_bias_linear:
            patterns.extend([
                r"\d+.self_attention.dense.bias",
                r"\d+.mlp.dense_4h_to_h.bias",
            ])
        if args.normalization == 'layernorm':
            patterns.extend([
                r"\d+.input_layernorm.bias",
                r"\d+.post_attention_layernorm.bias",
                r"\d+.bias",
            ])
        return patterns

    @staticmethod
    def _get_row_parallel_param_patterns():
        return [
            r"\d+.mlp.dense_4h_to_h.weight",
            r"\d+.self_attention.dense.weight",
        ]

    @staticmethod
    def _get_swiglu_col_parallel_param_patterns():
        args = get_args()
        if not args.swiglu:
            return []
        patterns = [
            r"\d+.mlp.dense_h_to_4h.weight",
        ]
        if args.add_bias_linear:
            patterns.append(r"\d+.mlp.dense_h_to_4h.bias")
        return patterns


    def universal_checkpoint_info(self):
        info = dict()
        if DS_UNIVERSAL_CHECKPOINT_INFO:
            # Vocabulary parameters (embeddings) that require special handling due to padding.
            info[VOCABULARY_PARAMETER_PATTERNS] = self._get_vocab_param_patterns()

            # Replicated (shared) parameters on the pipeline dimension
            info[PIPELINE_REPLICATED_PARAMETER_PATTERNS] = self._get_pp_replicated_param_patterns()

            # Parameter slices that should be averaged not concatenated.
            info[TP_REPLICATED_PARAMETER_PATTERNS] = self._get_tp_replicated_param_patterns()

            # Parameter that are sliced on the row dimension
            info[PARAMETER_WITH_ROW_PARALLELISM_PATTERNS] = self._get_row_parallel_param_patterns()

            # SWIGLU parameters are first sliced on dim=0 to tp slices
            # Then, each tp slice is chunked into 2 to create the linear layers L1, L2 used for silu(L1(x)) * L2(x))
            info[PARAMETER_WITH_2_SUB_PARAMS_CAT_DIM_0] = self._get_swiglu_col_parallel_param_patterns()
        return info
                         
                        
