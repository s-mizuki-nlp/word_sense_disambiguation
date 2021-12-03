#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import division, absolute_import, print_function, unicode_literals

import torch
from torch import nn

from model.encoder_internal import BaseHashCode
from model.hashembed import HashEmbedding


class HashCodeAwareEmbedding(BaseHashCode):

    def __init__(self, n_seq_len: int, num_embeddings: int, embedding_dim: int, num_buckets: int,
                 num_hashes: int = 2, n_prefix_hash_bins: int = int(1E14), pad_trailing_zeroes: bool = True,
                 append_weight: bool = True,
                 **kwargs):

        super().__init__(n_prefix_hash_bins=n_prefix_hash_bins, max_seq_len=n_seq_len,
                         pad_trailing_zeroes=pad_trailing_zeroes, random_seed=42, mask_zero=True)

        self.emb_layer = HashEmbedding(num_embeddings=num_embeddings, num_hashes=num_hashes,
                                       embedding_dim=embedding_dim - num_hashes if append_weight else embedding_dim,
                                       num_buckets=num_buckets, append_weight=append_weight,
                                       **kwargs)
        self.n_seq_len = n_seq_len

    def init_weights(self, *args, **kwargs):
        self.emb_layer.reset_parameters()

    def forward(self, input_sequence: torch.Tensor) -> torch.Tensor:
        # input_sequence: (n_batch, n_digits_so_far)
        assert input_sequence.ndim == 2, f"unexpected dimension size: {input_sequence.ndim}"

        # input_sequence_prefix_hashes: (n_batch, n_digits_so_far)
        input_sequence_prefix_hashes = self.transform_sequence_to_prefix_hashes(input_sequence)
        # t_emb: (n_batch, n_digits_so_far, n_emb)
        t_emb = self.emb_layer.forward(input_sequence_prefix_hashes)

        return t_emb


class PositionAwareEmbedding(torch.nn.Module):

    def __init__(self, n_seq_len: int = None, **kwargs):

        super().__init__()
        if isinstance(n_seq_len, int):
            lst_layers = [nn.Embedding(**kwargs) for _ in range(n_seq_len)]
            self.emb_layers = nn.ModuleList(lst_layers)
        else:
            self.emb_layers = nn.Embedding(**kwargs)
        self.n_seq_len = n_seq_len

    def init_weights(self, lower: float, upper: float):
        if isinstance(self.emb_layers, nn.ModuleList):
            for layer in self.emb_layers:
                nn.init.uniform_(layer.weight, lower, upper)
        else:
            nn.init.uniform_(self.emb_layers.weight, lower, upper)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 2, f"unexpected dimension size: {x.ndim}"
        if isinstance(self.emb_layers, nn.ModuleList):
            n_digits = x.shape[-1]
            lst_t_emb = [self.emb_layers[digit](x[:,digit]) for digit in range(n_digits)]
            t_emb = torch.stack(lst_t_emb, dim=1)
        else:
            t_emb = self.emb_layers.forward(x)
        return t_emb