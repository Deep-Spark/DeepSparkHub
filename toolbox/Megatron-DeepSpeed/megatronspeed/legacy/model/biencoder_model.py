import os
import torch
import sys

from megatron.training import get_args, print_rank_0, get_tokenizer
from megatron.core import mpu
from megatron.training.checkpointing import fix_query_key_value_ordering
from megatron.training.checkpointing import get_checkpoint_tracker_filename
from megatron.training.checkpointing import get_checkpoint_name
from megatron.legacy.model.bert_model import bert_position_ids
from megatron.legacy.model.enums import AttnMaskType
from megatron.legacy.model.language_model import get_language_model
from megatron.legacy.model.utils import get_linear_layer
from megatron.legacy.model.utils import init_method_normal
from megatron.legacy.model.utils import scaled_init_method_normal
from megatron.legacy.model.module import MegatronModule


class PretrainedBertModel(MegatronModule):
    """BERT-based encoder for queries or contexts used for
    learned information retrieval."""

    def __init__(self, num_tokentypes=2,
            parallel_output=True, pre_process=True, post_process=True):
        super(PretrainedBertModel, self).__init__()

        args = get_args()
        tokenizer = get_tokenizer()
        self.pad_id = tokenizer.pad
        self.biencoder_projection_dim = args.biencoder_projection_dim
        self.parallel_output = parallel_output
        self.pre_process = pre_process
        self.post_process = post_process
        init_method = init_method_normal(args.init_method_std)
        scaled_init_method = scaled_init_method_normal(
            args.init_method_std, args.num_layers)

        self.language_model, self._language_model_key = get_language_model(
            num_tokentypes=num_tokentypes,
            add_pooler=False,
            encoder_attn_mask_type=AttnMaskType.padding,
            init_method=init_method,
            scaled_init_method=scaled_init_method,
            pre_process=self.pre_process,
            post_process=self.post_process)

        if args.biencoder_projection_dim > 0:
            self.projection_enc = get_linear_layer(args.hidden_size,
                                                   args.biencoder_projection_dim,
                                                   init_method,
                                                   gather_params_on_init=args.zero_stage == 3)
            self._projection_enc_key = 'projection_enc'

    def forward(self, input_ids, attention_mask, tokentype_ids=None):
        extended_attention_mask = attention_mask.unsqueeze(1)
        #extended_attention_mask = bert_extended_attention_mask(attention_mask)
        position_ids = bert_position_ids(input_ids)

        lm_output = self.language_model(input_ids,
                                        position_ids,
                                        extended_attention_mask,
                                        tokentype_ids=tokentype_ids)
        # This mask will be used in average-pooling and max-pooling
        pool_mask = (input_ids == self.pad_id).unsqueeze(2)

        # Taking the representation of the [CLS] token of BERT
        pooled_output = lm_output[0, :, :]

        # Converting to float16 dtype
        pooled_output = pooled_output.to(lm_output.dtype)

        # Output.
        if self.biencoder_projection_dim:
            pooled_output = self.projection_enc(pooled_output)

        return pooled_output

    def state_dict_for_save_checkpoint(self, prefix='', keep_vars=False):
        """For easy load when model is combined with other heads,
        add an extra key."""

        state_dict_ = {}
        state_dict_[self._language_model_key] \
            = self.language_model.state_dict_for_save_checkpoint(
                prefix=prefix, keep_vars=keep_vars)

        if self.biencoder_projection_dim > 0:
            state_dict_[self._projection_enc_key] = \
                self.projection_enc.state_dict(prefix=prefix,
                                               keep_vars=keep_vars)

        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        """Customized load."""
        print_rank_0("loading pretrained weights")
        self.language_model.load_state_dict(
            state_dict[self._language_model_key], strict=strict)

        if self.biencoder_projection_dim > 0:
            print_rank_0("loading projection head weights")
            self.projection_enc.load_state_dict(
                state_dict[self._projection_enc_key], strict=strict)
