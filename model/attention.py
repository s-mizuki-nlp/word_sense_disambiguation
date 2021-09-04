#!/usr/bin/env python
# -*- coding:utf-8 -*-

from typing import Optional, Dict, Any, Union, Tuple
import torch
from torch.nn import MultiheadAttention

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
