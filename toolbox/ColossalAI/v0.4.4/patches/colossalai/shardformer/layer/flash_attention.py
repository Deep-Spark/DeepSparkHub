import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.cache_utils import Cache
from transformers.utils import logging

from colossalai.lazy import LazyInitContext

logger = logging.get_logger(__name__)

from transformers.utils import (
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
)
if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa
    from flash_attn.flash_attn_interface import flash_attn_varlen_func as flash_attn_unpadded_func
    from flash_attn.flash_attn_interface import flash_attn_func
from transformers.models.llama.modeling_llama import LlamaAttention
from .rotary_pos_embedding import RotaryEmbedding

# from apex.transformer.functional import fused_apply_rotary_pos_emb
from ixformer.train import fused_apply_split_rotary_pos_emb

try:
    from einops import rearrange
except ImportError:
    rearrange = None

from transformers.models.llama.configuration_llama import LlamaConfig
from colossalai.shardformer.layer import LinearWithFusedGradientAccu

def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class FlashSelfAttentionCore(torch.nn.Module):
    """Implement the scaled dot product attention with softmax.
    Arguments
    ---------
        softmax_scale: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.0)
    """
    def __init__(self, causal=False, softmax_scale=None, attention_dropout=0.0,
                 device=None, dtype=None):
        super().__init__()
        assert flash_attn_unpadded_func is not None, ('Please install FlashAttention first, '
                                                      'e.g., with pip install flash-attn')
        assert rearrange is not None, 'Please install einops first, e.g., with pip install einops'
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.dropout_p = attention_dropout

    def forward(self, q, k, v):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            q, k, v: The tensor containing the query, key, and value. (B, S, H, D)
        """

        assert all((i.dtype in [torch.float16, torch.bfloat16] for i in (q,k,v)))
        assert all((i.is_cuda for i in (q,k,v)))

        batch_size, seqlen_q = q.shape[0], q.shape[1]
        seqlen_k = k.shape[1]

        # q, k, v = [rearrange(x, 'b s ... -> (b s) ...') for x in [q, k, v]]
        # # if os.getenv('ENABLE_FLASH_ATTENTION_WITH_IXDNN', '0') != '0':
        # #     cu_seqlens_q = torch.empty((batch_size), dtype=torch.int32, device=q.device)
        # # else:
        # cu_seqlens_q = torch.arange(0, (batch_size + 1) * seqlen_q, step=seqlen_q, dtype=torch.int32,
        #                                 device=q.device)

        # if self.training:
        #     # during training q,k,v always have same seqlen
        #     assert seqlen_k == seqlen_q

        #     is_causal = self.causal
        #     cu_seqlens_k = cu_seqlens_q
        #     dropout_p = self.dropout_p
        # else:
        #     # turn off FA causal mask after first inference autoregressive iteration
        #     # only on first autoregressive step q,k,v have same seqlen
        #     is_causal = seqlen_q == seqlen_k
        #     cu_seqlens_k = torch.arange(0, (batch_size + 1) * seqlen_k, step=seqlen_k, dtype=torch.int32,
        #                 device=q.device)
        #     dropout_p = 0

        # output = flash_attn_unpadded_func(
        #     q, k, v, cu_seqlens_q, cu_seqlens_k, seqlen_q, seqlen_k,
        #     dropout_p,
        #     softmax_scale=self.softmax_scale, causal=is_causal
        # )
        # output = rearrange(output, '(b s) ... -> b s ...', b=batch_size)


        self.attn_impl_mode = 1
        self.use_alibi = False
        self.alibi_mode = 1
        output = flash_attn_func(
            q,
            k,
            v,
            dropout_p=self.dropout_p,
            softmax_scale=self.softmax_scale,
            causal= self.causal,
            use_alibi=self.use_alibi,
            alibi_mode=self.alibi_mode,
            imp_mode=self.attn_impl_mode,
        )
        #output [b,s,h,d]

        return output


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        # self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        # self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        # self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        # self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.attention_bias)
        self.query_key_value = LinearWithFusedGradientAccu(self.hidden_size, self.head_dim * (self.num_heads + self.num_key_value_heads * 2), bias=config.attention_bias)
        self.o_proj = LinearWithFusedGradientAccu(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)

        # # partial rotary embeddings, which is better than full rotary
        # # Wang and Komatsuzaki et al
        # # https://github.com/kingoflolz/mesh-transformer-jax/
        rotary_dim = config.hidden_size // config.num_attention_heads
        rotary_pos_emb = RotaryEmbedding(
            rotary_dim,
            rotary_percent = 1,
            seq_len_interpolation_factor = 1,
            rotary_base=config.rope_theta
        )
        self.rotary_pos_emb = rotary_pos_emb(config.max_position_embeddings)
        self.rotary_pos_emb = ((self.rotary_pos_emb,) * 2)

        # self._init_rope()

    # def _init_rope(self):
    #     if self.config.rope_scaling is None:
    #         self.rotary_emb = LlamaRotaryEmbedding(
    #             self.head_dim,
    #             max_position_embeddings=self.max_position_embeddings,
    #             base=self.rope_theta,
    #         )
    #     else:
    #         scaling_type = self.config.rope_scaling["type"]
    #         scaling_factor = self.config.rope_scaling["factor"]
    #         if scaling_type == "linear":
    #             self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
    #                 self.head_dim,
    #                 max_position_embeddings=self.max_position_embeddings,
    #                 scaling_factor=scaling_factor,
    #                 base=self.rope_theta,
    #             )
    #         elif scaling_type == "dynamic":
    #             self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
    #                 self.head_dim,
    #                 max_position_embeddings=self.max_position_embeddings,
    #                 scaling_factor=scaling_factor,
    #                 base=self.rope_theta,
    #             )
    #         else:
    #             raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        return None
    
class Colo_LlamaFlashAttention2(LlamaAttention):
    """
    基于 transformers (v4.39.3) LlamaFlashAttention2 改进, 优化点：
    a. 融合 self.q_proj、self.k_proj、self.v_proj 为 self.query_key_value ；
    b. self.query_key_value 和 self.o_proj 改为 LinearWithFusedGradientAccu 类型 ；
    c. 融合 split_mixed_q_k_v 和 rope ；
    d. self_attention core 的实现改为 core_attention_flash，内部使用定长的 flash_attn_func ；

    
    Llama flash attention module. This module inherits from `LlamaAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

        self.core_attention_flash = FlashSelfAttentionCore(
            causal=True, attention_dropout=self.attention_dropout
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        output_attentions = False

        bsz, q_len, _ = hidden_states.size()

        hidden_states = hidden_states.transpose(0,1).contiguous()
        mixed_x_layer = self.query_key_value(hidden_states)
        # [sq, b, ng * (np/ng + 2) * hn] --> [sq, b, ng, (np/ng + 2), hn]
        new_tensor_shape = (mixed_x_layer.size()[0], mixed_x_layer.size()[1],
            self.num_heads,
            (1 + 2),
            self.head_dim,
        )
        mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

        rotary_pos_emb = self.rotary_pos_emb[0]
        query_states, key_states, value_states = fused_apply_split_rotary_pos_emb(
            mixed_x_layer,
            rotary_pos_emb,
        )
        query_states, key_states, value_states = [rearrange(x, "s b h d -> b s h d").contiguous() for x in (query_states, key_states, value_states)]


        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        # attn_output = self._flash_attention_forward(
        #     query_states, key_states, value_states, attention_mask, q_len, dropout=dropout_rate
        # )
        attn_output = self.core_attention_flash(query_states, key_states, value_states)

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def _flash_attention_forward(
        self, query_states, key_states, value_states, attention_mask, query_length, dropout=0.0, softmax_scale=None
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`float`):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        """
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1. For details, please see the comment in LlamaFlashAttention2 __init__.
            causal = self.is_causal and query_length != 1

        # Contains at least one padding token in the sequence
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=causal,
            )

            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            attn_output = flash_attn_func(
                query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=causal
            )

        return attn_output

    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )


class Colo_LlamaFlashAtten(Colo_LlamaFlashAttention2):
    def __init__(self) -> None:
        raise NotImplementedError(
            "Colo_LlamaFlashAtten is not implemented as a physical class. "
            "It is meant to be used only with the from_native_module interface to Convert a native LlamaFlashAttention2(from transformers) module to Colo_LlamaFlashAttention2 module provided above."
        )
        
    @staticmethod
    def from_native_module(module: nn.Module, *args, **kwargs) -> nn.Module:
        
        LazyInitContext.materialize(module)
        
        config = getattr(module, "config")
        layer_idx = getattr(module, "layer_idx")

        flash_atten = Colo_LlamaFlashAttention2(config=config, layer_idx=layer_idx)
        
        return flash_atten
