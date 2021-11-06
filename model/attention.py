#!/usr/bin/env python
# -*- coding:utf-8 -*-

from typing import Optional, Dict, Any, Union, Tuple
import torch
from torch import nn
from torch.nn import MultiheadAttention, Linear

from .utils import masked_average


class EntityVectorEncoder(MultiheadAttention):

    def __init__(self, embed_dim, num_heads, q_input_feature: str = "entity", kv_input_feature: str = "context",
                 dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None,
                 batch_first: bool = True, average_pooling: bool = True):
        super().__init__(embed_dim, num_heads, dropout, bias, add_bias_kv, add_zero_attn, kdim, vdim)

        set_features = {"entity","context"}
        if kv_input_feature not in set_features:
            raise ValueError(f"`kv_input_feature` must be: {set_features}")
        if q_input_feature not in set_features:
            raise ValueError(f"`q_input_feature` must be: {set_features}")

        self._q_input_feature = q_input_feature
        self._kv_input_feature = kv_input_feature
        self._batch_first = batch_first
        self._average_pooling = average_pooling

    def forward(self, entity_embeddings: torch.Tensor, context_embeddings: torch.Tensor,
                entity_sequence_mask: torch.BoolTensor, context_sequence_mask: torch.BoolTensor) -> torch.Tensor:
        """
        calculate entity vectors using Multihead attention and 1D pooling.
        entity vector = 1DPooling( MHA(Q=Entity, K=V=Context) )

        @param entity_embeddings: embeddings of entity subwords. (n_batch, max(n_entity_seq_len), n_dim)
        @param context_embeddings: embeddings of whole-sentence subwords. (n_batch, max(n_seq_len), n_dim)
        @param entity_sequence_mask: entity mask. (n_batch, max(n_entity_seq_len))
        @param context_sequence_mask: context mask. (n_batch, max(n_seq_len))
        @return: entity vectors. (n_batch, n_dim)
        """

        # swap batch dimension: (n_batch, n_seq_len, n_dim) -> (n_seq_len, n_batch, n_dim)
        if self._batch_first:
            entity_embeddings = entity_embeddings.swapaxes(0,1)
            context_embeddings = context_embeddings.swapaxes(0,1)

        # query
        if self._q_input_feature == "context":
            t_query = context_embeddings
            t_q_padding_mask = context_sequence_mask
        elif self._q_input_feature == "entity":
            t_query = entity_embeddings
            t_q_padding_mask = entity_sequence_mask
        else:
            raise ValueError(f"unknown argument: {self._q_input_feature}")

        if self._kv_input_feature == "context":
            t_key_value = context_embeddings
            t_kv_padding_mask = context_sequence_mask
        elif self._kv_input_feature == "entity":
            t_key_value = entity_embeddings
            t_kv_padding_mask = entity_sequence_mask
        else:
            raise ValueError(f"unknown argument: {self._kv_input_feature}")

        # MHA
        t_output, _ = super().forward(query=t_query, key=t_key_value, value=t_key_value,
                                      key_padding_mask=t_kv_padding_mask, need_weights=False)
        if self._batch_first:
            t_output = t_output.swapaxes(0,1)

        # 1D pooling
        if self._average_pooling:
            t_encoded = masked_average(embeddings=t_output, sequence_mask=t_q_padding_mask)
        else:
            t_encoded = t_output

        return t_encoded

    def summary(self):
        ret = {
            "class_name":self.__class__.__name__,
            "num_heads":self.num_heads,
            "embed_dim":self.embed_dim,
            "q_input_feature": self._q_input_feature,
            "kv_input_feature": self._kv_input_feature
        }
        return ret


class InitialStatesEncoder(EntityVectorEncoder):

    def __init__(self, embed_dim: int, num_heads: int, q_input_feature: str = "entity", kv_input_feature: str = "context",
                 dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None,
                 initial_state_dim=None, batch_first: bool = True):

        super().__init__(embed_dim, num_heads, q_input_feature, kv_input_feature, dropout,
                         bias, add_bias_kv, add_zero_attn, kdim, vdim, batch_first)

        initial_state_dim = embed_dim if initial_state_dim is None else initial_state_dim
        self._activation = nn.GELU()
        self._v_to_h = Linear(in_features=embed_dim, out_features=initial_state_dim)
        self._v_to_c = Linear(in_features=embed_dim, out_features=initial_state_dim)

    def forward(self, entity_embeddings: torch.Tensor, context_embeddings: torch.Tensor,
                entity_sequence_mask: torch.BoolTensor, context_sequence_mask: torch.BoolTensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        calculate initial state vector pairs of LSTM using Multihead attention, 1D pooling, and Linear Transformation.
        v = 1DPooling( MHA(Q=Entity, K=V=Context or Entity) )
        h = Linear(v)
        c = Linear(v)

        @param entity_embeddings: embeddings of entity subwords. (n_batch, max(n_entity_seq_len), n_dim)
        @param context_embeddings: embeddings of whole-sentence subwords. (n_batch, max(n_seq_len), n_dim)
        @param entity_sequence_mask: entity mask. (n_batch, max(n_entity_seq_len))
        @param context_sequence_mask: context mask. (n_batch, max(n_seq_len))
        @return: initial state vectors. Tuple[(n_batch, n_initial_state_dim), (n_batch, n_initial_state_dim)]
        """

        # t_v: (n_batch, n_dim)
        t_v_dash = super().forward(entity_embeddings, context_embeddings, entity_sequence_mask, context_sequence_mask)
        t_v = self._activation(t_v_dash)
        t_h = self._v_to_h(t_v)
        t_c = self._v_to_c(t_v)

        return (t_h, t_c)

    def summary(self):
        ret = super().summary()
        ret["class_name"] = self.__class__.__name__
        return ret


class InitialStatesEncoderAveragePooling(nn.Module):

    def __init__(self, embed_dim, initial_state_dim=None):
        super().__init__()

        self.embed_dim = embed_dim
        self.initial_state_dim = embed_dim if initial_state_dim is None else initial_state_dim
        self._v_to_h = Linear(in_features=self.embed_dim, out_features=self.initial_state_dim)
        self._v_to_c = Linear(in_features=self.embed_dim, out_features=self.initial_state_dim)

    def forward(self, entity_embeddings: torch.Tensor,
                entity_sequence_mask: torch.BoolTensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        calculate initial state vector pairs of LSTM using 1D pooling and Linear Transformation.
        v = 1DPooling( entity vectors )
        h = Linear(v)
        c = Linear(v)

        @param entity_embeddings: embeddings of entity subwords. (n_batch, max(n_entity_seq_len), n_dim)
        @param context_embeddings: embeddings of whole-sentence subwords. (n_batch, max(n_seq_len), n_dim)
        @param entity_sequence_mask: entity mask. (n_batch, max(n_entity_seq_len))
        @param context_sequence_mask: context mask. (n_batch, max(n_seq_len))
        @return: initial state vectors. Tuple[(n_batch, n_initial_state_dim), (n_batch, n_initial_state_dim)]
        """

        # t_v: (n_batch, n_dim)
        t_v = masked_average(embeddings=entity_embeddings, sequence_mask=entity_sequence_mask)
        t_h = self._v_to_h(t_v)
        t_c = self._v_to_c(t_v)

        return (t_h, t_c)

    def summary(self):
        ret = {
            "class_name":self.__class__.__name__,
            "embed_dim":self.embed_dim
        }
        return ret