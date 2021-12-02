#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

from typing import Optional

import warnings
import torch
from torch import nn
import math


class HashEmbeddingsLogits(torch.nn.Module):

    def __init__(self, n_digits: int, n_prefix: int, n_ary_out: int, n_dim_emb: int,
                 dimensionality_reduction: bool = True,
                 **kwargs):

        super().__init__()


class AdditiveCodeAwareLogits(torch.nn.Module):

    def __init__(self, n_digits: int, n_ary_in: int, n_ary_out: int, n_dim_emb: int,
                 bias: bool = True,
                 depends_on_previous_digits: Optional[int] = None,
                 **kwargs):

        super().__init__()
        self._n_digits = n_digits
        self._n_ary_in = n_ary_in
        self._n_ary_out = n_ary_out
        self._n_dim_emb = n_dim_emb
        self._bias = bias
        self._depends_on_previous_digits = depends_on_previous_digits

        cfg_base_weight_layer = {
            "num_embeddings": n_ary_in,
            "embedding_dim": n_ary_out * n_dim_emb
        }
        cfg_base_weight_layer.update(kwargs)

        # base_weight_layers: (n_digit, n_ary_in, n_ary_out * n_dim)
        lst_base_weight_layers = [nn.Embedding(**cfg_base_weight_layer) for _ in range(n_digits)]
        self.base_weight_layers = nn.ModuleList(lst_base_weight_layers)

        # offset_weights: (n_digit, n_ary_out * n_dim)
        if bias:
            self.bias_weights = nn.Parameter(torch.zeros(size=(n_digits, n_ary_out * n_dim_emb)), requires_grad=True)

        self.init_weights()

    def init_weights(self):
        for layer in self.base_weight_layers:
            nn.init.zeros_(layer.weight)

    def _ragged_cumsum(self, tensor: torch.Tensor, dim: int, stride: Optional[int]):
        if stride is None:
            # t_cumsum[:,d,:] = tensor[:,:d+1,:].sum(dim=dim)
            t_cumsum = torch.cumsum(tensor, dim=dim)
        else:
            # t_cumsum[:,d,:] = tensor[:,(d-stride):d+1,:].sum(dim=dim)
            shp = list(tensor.shape)
            length = shp[dim]

            _stride = min(stride + 1, length)
            t_ = torch.cumsum(tensor, dim=dim)

            shp[dim] = _stride
            pad = torch.zeros(shp, dtype=tensor.dtype).to(tensor.device)
            index = torch.arange(end=length - _stride).to(tensor.device)
            t_ragged = torch.index_select(t_, dim=dim, index=index)

            t_cumsum = t_ - torch.cat((pad, t_ragged), dim=dim)

        return t_cumsum

    def forward(self, input_sequence: torch.Tensor, t_representation: torch.Tensor):
        # input_sequence: (n_batch, n_digits_so_far) input_sequence[b,d] \in {0,n_ary_in}
        # t_representation: (n_batch, n_digits_so_far, n_dim)

        n_digits_so_far = min(self._n_digits, input_sequence.shape[-1])
        lst_base_weights = [self.base_weight_layers[digit](input_sequence[:,digit]) for digit in range(n_digits_so_far)]
        # t_base_weight: (n_batch, n_digits_so_far, n_ary_out * n_dim)
        t_base_weight = torch.stack(lst_base_weights, dim=1)
        if self._depends_on_previous_digits is None:
            t_weight_ = torch.cumsum(t_base_weight, dim=1)
        else:
            t_weight_ = self._ragged_cumsum(t_base_weight, dim=1, stride=min(self._depends_on_previous_digits, n_digits_so_far))
        if self._bias:
            t_weight_ = t_weight_ + self.bias_weights[:n_digits_so_far, :]
        # t_weight: (n_batch, n_digits_so_far, n_ary_out, n_dim)
        t_weight = t_weight_.view((-1, n_digits_so_far, self._n_ary_out, self._n_dim_emb))
        # t_logits: (n_batch, n_digits_so_far, n_ary_out)
        t_logits = torch.matmul(t_weight, t_representation.unsqueeze(-1)).squeeze(-1)

        return t_logits


class PositionAwareLogits(torch.nn.Module):

    def __init__(self, n_seq_len: int = None, **kwargs):

        super().__init__()
        if isinstance(n_seq_len, int):
            lst_layers = [nn.Linear(**kwargs) for _ in range(n_seq_len)]
            self.linear_layers = nn.ModuleList(lst_layers)
        else:
            self.linear_layers = nn.Linear(**kwargs)
        self.n_seq_len = n_seq_len

        self.init_weights()

    def init_weights(self):
        if isinstance(self.linear_layers, nn.ModuleList):
            for layer in self.linear_layers:
                nn.init.zeros_(layer.weight)
        else:
            nn.init.zeros_(self.linear_layers.weight)

    def forward(self, t_representation: torch.Tensor, **kwargs) -> torch.Tensor:
        # t_representation: (n_batch, n_digits_so_far, n_dim)
        assert t_representation.ndim == 3, f"unexpected dimension size: {t_representation.ndim}"
        if isinstance(self.linear_layers, nn.ModuleList):
            n_digits = t_representation.shape[1]
            lst_t_logits = [self.linear_layers[digit](t_representation[:,digit,:]) for digit in range(n_digits)]
            t_logits = torch.stack(lst_t_logits, dim=1)
        else:
            t_logits = self.linear_layers.forward(t_representation)
        # t_logits: (n_batch, n_digits_so_far, n_ary)
        return t_logits


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
