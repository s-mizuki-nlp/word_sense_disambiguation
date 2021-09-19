#!/usr/bin/env python
# -*- coding:utf-8 -*-

from typing import Optional, Dict, Any, Union, Tuple
import torch
from torch.nn import MultiheadAttention, Linear

from .utils import masked_average


class EntityVectorEncoder(MultiheadAttention):

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
        t_query = entity_embeddings.swapaxes(0,1)
        t_key_value = context_embeddings.swapaxes(0,1)

        # MHA
        entity_embeddings, _ = super().forward(query=t_query, key=t_key_value, value=t_key_value,
                                      key_padding_mask=context_sequence_mask, need_weights=False)
        entity_embeddings = entity_embeddings.swapaxes(0,1)

        # 1D pooling
        entity_vectors = masked_average(embeddings=entity_embeddings, sequence_mask=entity_sequence_mask)

        return entity_vectors


class InitialStatesEncoder(EntityVectorEncoder):

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None, initial_state_dim=None):

        super().__init__(embed_dim, num_heads, dropout, bias, add_bias_kv, add_zero_attn, kdim, vdim)

        initial_state_dim = embed_dim if initial_state_dim is None else initial_state_dim
        self._v_to_h = Linear(in_features=embed_dim, out_features=initial_state_dim)
        self._v_to_c = Linear(in_features=embed_dim, out_features=initial_state_dim)

    def forward(self, entity_embeddings: torch.Tensor, context_embeddings: torch.Tensor,
                entity_sequence_mask: torch.BoolTensor, context_sequence_mask: torch.BoolTensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        calculate initial state vector pairs of LSTM using Multihead attention, 1D pooling, and Linear Transformation.
        v = 1DPooling( MHA(Q=Entity, K=V=Context) )
        h = Linear(v)
        c = Linear(v)

        @param entity_embeddings: embeddings of entity subwords. (n_batch, max(n_entity_seq_len), n_dim)
        @param context_embeddings: embeddings of whole-sentence subwords. (n_batch, max(n_seq_len), n_dim)
        @param entity_sequence_mask: entity mask. (n_batch, max(n_entity_seq_len))
        @param context_sequence_mask: context mask. (n_batch, max(n_seq_len))
        @return: initial state vectors. Tuple[(n_batch, n_initial_state_dim), (n_batch, n_initial_state_dim)]
        """

        # t_v: (n_batch, n_dim)
        t_v = super().forward(entity_embeddings, context_embeddings, entity_sequence_mask, context_sequence_mask)
        t_h = self._v_to_h(t_v)
        t_c = self._v_to_c(t_v)

        return (t_h, t_c)
