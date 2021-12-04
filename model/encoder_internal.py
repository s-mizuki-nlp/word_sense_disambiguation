#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

from typing import Dict, Optional, List, Iterable, Union

import torch
from torch import nn
import math

from dataset.utils import sequence_to_str
from .hashembed.embedding import HashFamily
from collections import Counter, defaultdict

def min_excluding_zeroes(values: Iterable[int], default_value=0):
    def is_natural_number(v: int):
        return v > 0

    return min(filter(is_natural_number, values), default=default_value)


class BasePrefixAwareLayer(torch.nn.Module):

    def __init__(self, replace_trailing_zeroes: bool,
                 num_classes: Optional[int] = None, null_prefix_index: Optional[int] = None,
                 unobserved_class_fill_strategy: Union[str, int] = "min",
                 smoothing_alpha: float = 0.1):
        """

        @param replace_trailing_zeroes: fill trailing prefix index using last non-zero index.
        @param num_classes: number of classes used for prior probability setup.
        @param null_prefix_index: prefix index used for undefined prefix.
        @param smoothing_alpha: smoothing parameter used for calculating prior probability: \pi(c) = count(c) + \alpha / \sum_{c'}{count(c')+\alpha}
        @param unobserved_class_fill_strategy:
        @param num_classes: number of classes. i.e., n_ary
        @param smoothing_alpha: smoothing parameter for zero-freq entry.
        """
        super().__init__()
        self._num_classes = num_classes
        self._null_prefix_index = null_prefix_index
        self._unobserved_class_fill_strategy = unobserved_class_fill_strategy
        self._replace_trailing_zeroes = replace_trailing_zeroes
        self._sense_code_prefix_index = None
        self._sense_code_prefix_statistics = None
        self._smoothing_alpha = smoothing_alpha

    def transform_sequence_to_prefix_indices(self, sequences: torch.Tensor) -> torch.Tensor:
        """
        this function transform sense codes to the prefix ids of sense codes.
        then, optionally, fill zeroes with last hash values.
        e.g.,
            hash([[1,2,0]]) -> [[hash(1), hash([1,2]), 0]] if fill_trailing_zeroes = False
            hash([[1,2,0]]) -> [[hash(1), hash([1,2]), hash([1,2])]] if fill_trailing_zeroes = True

        @param sequences: sequence to be prefix-indexed.
        """

        # sequence_inputs: (n_batch, n_seq_len)
        # prefix_ids: (n_batch, n_seq_len)
        device = sequences.device
        n_seq_len = sequences.shape[-1]
        lst_prefix_indices = []
        for sequence in sequences.tolist():
            code_length = n_seq_len - sequence.count(0)
            str_prefixes = [sequence_to_str(sequence[:d]) for d in range(1, code_length+1)]
            prefix_indices = [self.sense_code_prefix_index.get(prefix, self._null_prefix_index) for prefix in str_prefixes]

            if self._replace_trailing_zeroes:
                fill_value = prefix_indices[-1]
            else:
                fill_value = 0
            # fill trailing digits
            prefix_indices = prefix_indices + [fill_value] * (n_seq_len - code_length)
            lst_prefix_indices.append(prefix_indices)

        t_prefix_indices = torch.LongTensor(lst_prefix_indices).to(device)

        return t_prefix_indices

    def get_prior_probabilities(self, sequences: torch.Tensor) -> torch.Tensor:
        """
        returns class-prior probability of each position in the sequence based on prefix statistics.

        @param sequences: (n_batch, n_digits)
        @return: (n_batch, n_digits, num_classes)
        """
        device = sequences.device
        lst_lst_counts = [] # List[List[List[float]]]
        sequence_prefix_indices = self.transform_sequence_to_prefix_indices(sequences)
        for prefix_indices in sequence_prefix_indices.tolist():
            lst_counts = [self.get_sense_code_prefix_counts(index) for index in prefix_indices]
            lst_lst_counts.append(lst_counts)

        t_prior = torch.Tensor(lst_lst_counts, device=device)
        # normalize for each class
        t_prior = t_prior / t_prior.sum(dim=-1, keepdim=True)

        return t_prior

    def get_sense_code_prefix_counts(self, prefix_index: int) -> List[float]:
        # smoothing_alpha is used for prior of prior.
        lst_counts = [self._smoothing_alpha] * self._num_classes

        lookup_key = prefix_index
        if lookup_key not in self._sense_code_prefix_statistics:
            return lst_counts

        dict_stats = self._sense_code_prefix_statistics[lookup_key]
        if len(dict_stats) == 0:
            return lst_counts

        if isinstance(self._unobserved_class_fill_strategy, int):
            fill_value = self._unobserved_class_fill_strategy
        elif self._unobserved_class_fill_strategy == "min":
            fill_value = min_excluding_zeroes(dict_stats.values(), default_value=0)
        else:
            raise AttributeError(f"unknown `unobserved_class_fill_strategy` value: {self._unobserved_class_fill_strategy}")

        for idx, count in dict_stats.items():
            # fill unobserved
            count = count if count > 0 else fill_value
            lst_counts[idx] += count

        return lst_counts

    @property
    def sense_code_prefix_index(self) -> Dict[str, int]:
        return self._sense_code_prefix_index

    @sense_code_prefix_index.setter
    def sense_code_prefix_index(self, new_value):
        assert self._sense_code_prefix_index is None, f"prefix index is already set. this attribute is single write only."
        self._sense_code_prefix_index = new_value

    @property
    def sense_code_prefix_statistics(self) -> Dict[int, Dict[int, int]]:
        return self._sense_code_prefix_statistics

    @sense_code_prefix_statistics.setter
    def sense_code_prefix_statistics(self, new_value: Dict[int, Dict[int, int]]):
        assert self._sense_code_prefix_statistics is None, f"prefix statistics is already set. this attribute is single write only."
        self._sense_code_prefix_statistics = new_value


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
