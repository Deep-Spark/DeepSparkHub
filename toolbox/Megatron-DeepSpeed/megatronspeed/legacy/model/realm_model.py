from functools import wraps

import torch

from megatron.training import get_args, print_rank_0
from megatron.training.checkpointing import get_checkpoint_tracker_filename, get_checkpoint_name
from megatron.legacy.model import BertModel
from megatron.legacy.model.module import MegatronModule
from megatron.core import mpu
from megatron.legacy.model.enums import AttnMaskType
from megatron.legacy.model.utils import get_linear_layer
from megatron.legacy.model.utils import init_method_normal
from megatron.legacy.model.language_model import get_language_model
from megatron.legacy.model.utils import scaled_init_method_normal
from megatron.legacy.model.bert_model import bert_extended_attention_mask, bert_position_ids

class IREncoderBertModel(MegatronModule):
    """BERT-based encoder for queries or blocks used for learned information retrieval."""
    def __init__(self, ict_head_size, num_tokentypes=2, parallel_output=True):
        super(IREncoderBertModel, self).__init__()
        args = get_args()

        self.ict_head_size = ict_head_size
        self.parallel_output = parallel_output
        init_method = init_method_normal(args.init_method_std)
        scaled_init_method = scaled_init_method_normal(args.init_method_std,
                                                       args.num_layers)

        self.language_model, self._language_model_key = get_language_model(
            num_tokentypes=num_tokentypes,
            add_pooler=True,
            encoder_attn_mask_type=AttnMaskType.padding,
            init_method=init_method,
            scaled_init_method=scaled_init_method)

        self.ict_head = get_linear_layer(args.hidden_size, ict_head_size, init_method, gather_params_on_init=args.zero_stage == 3)
        self._ict_head_key = 'ict_head'

    def forward(self, input_ids, attention_mask, tokentype_ids=None):
        extended_attention_mask = bert_extended_attention_mask(
            attention_mask, next(self.language_model.parameters()).dtype)
        position_ids = bert_position_ids(input_ids)

        lm_output, pooled_output = self.language_model(
            input_ids,
            position_ids,
            extended_attention_mask,
            tokentype_ids=tokentype_ids)

        # Output.
        ict_logits = self.ict_head(pooled_output)
        return ict_logits, None

    def state_dict_for_save_checkpoint(self, prefix='', keep_vars=False):
        """For easy load when model is combined with other heads,
        add an extra key."""

        state_dict_ = {}
        state_dict_[self._language_model_key] \
            = self.language_model.state_dict_for_save_checkpoint(prefix=prefix,
                                                                 keep_vars=keep_vars)
        state_dict_[self._ict_head_key] \
            = self.ict_head.state_dict(prefix=prefix,
                                       keep_vars=keep_vars)
        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        """Customized load."""
        self.language_model.load_state_dict(
            state_dict[self._language_model_key], strict=strict)
        self.ict_head.load_state_dict(
            state_dict[self._ict_head_key], strict=strict)
