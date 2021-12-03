#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import torch
from torch import nn
import math

from .hashembed.embedding import HashFamily


class BaseHashCode(torch.nn.Module):

    def __init__(self, n_prefix_hash_bins: int, max_seq_len: int, pad_trailing_zeroes: bool, random_seed: int = 42, mask_zero: bool=True):
        super().__init__()

        self._sequence_hash_function = HashFamily(bins=n_prefix_hash_bins, mask_zero=mask_zero, is_sequence_input=True,
                                                  max_seq_len=max_seq_len, random_seed=random_seed).draw_hash()
        self._pad_trailing_zeroes = pad_trailing_zeroes

    def transform_sequence_to_prefix_hashes(self, sequences: torch.Tensor):
        """
        this function transform sense codes to the prefix ids of sense codes.
        then, optionally, pads zeroes with last hash values.
        e.g.,
            hash([[1,2,0]]) -> [[hash(1), hash([1,2]), hash([1,2,0])]] if pad = False
            hash([[1,2,0]]) -> [[hash(1), hash([1,2]), hash([1,2])]] if pad = True
        by using fixed random seed and bins, you can obtain identical hash function every time.

        @param sequences: sequence to be hashed.
        """

        # sequence_inputs: (n_batch, n_digits)
        # prefix_ids: (n_batch, n_digits)
        n_digits = sequences.shape[-1]
        prefix_ids = self._sequence_hash_function(sequences)

        if self._pad_trailing_zeroes:
            # pad trailing zeroes with last prefix ids.
            n_code_lengths = (sequences != 0).sum(dim=-1)
            for idx, code_length in enumerate(n_code_lengths):
                if code_length >= n_digits:
                    continue
                prefix_ids[idx, code_length:] = prefix_ids[idx, code_length-1]

        return prefix_ids


class PositionalEncoding(torch.nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000, trainable: bool = False):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        if trainable:
            self.pe = nn.Parameter(torch.full(size=(1, max_len, d_model), fill_value=0.001), requires_grad=True)
        else:
            position = torch.arange(max_len).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
            pe = torch.zeros(1, max_len, d_model, requires_grad=False)
            pe[0, :, 0::2] = torch.sin(position * div_term)
            pe[0, :, 1::2] = torch.cos(position * div_term)
            self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[0, :x.shape[1], :]
        return self.dropout(x)
