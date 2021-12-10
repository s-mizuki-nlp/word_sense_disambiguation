#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import division, absolute_import, print_function, unicode_literals

from typing import Optional, Union

import torch
from torch import nn

from model.encoder_internal import BasePrefixAwareLayer, BaseLogitAdjustableLayer
from model.hashembed import HashEmbedding


class HashCodeAwareLogits(BaseLogitAdjustableLayer):

    def __init__(self, n_digits: int, n_ary_out: int,
                 num_embeddings: int, embedding_dim: int, num_buckets: int,
                 additive: bool,
                 logit_adjustment: bool,
                 num_hashes=2,
                 replace_trailing_zeroes: bool = False,
                 append_weight: bool = False,
                 **kwargs):

        if logit_adjustment:
            for required_argument in ("logit_adjust_tau", "logit_adjust_when"):
                assert required_argument in kwargs, f"argument {required_argument} must be specified."
            super().__init__(replace_trailing_zeroes=replace_trailing_zeroes, null_prefix_index=0,
                             num_classes=n_ary_out, unobserved_class_fill_strategy=kwargs.get("unobserved_class_fill_strategy", "min"),
                             smoothing_alpha=kwargs.get("smoothing_alpha", 0.1),
                             logit_adjust_when=kwargs["logit_adjust_when"],
                             logit_adjust_tau=kwargs["logit_adjust_tau"])
        else:
            super().__init__(replace_trailing_zeroes=replace_trailing_zeroes, null_prefix_index=0,
                             logit_adjust_when=False)

        self._n_digits = n_digits
        self._n_ary = n_ary_out
        self._n_dim_emb = embedding_dim
        self._n_distinc_prefix = num_embeddings
        self._logit_adjustment = logit_adjustment
        self._additive = additive

        # prefix hash から HashEmbeddingsを使って n_ary * n_dim_emb 個のparameterをlookupする
        self._logit_layer_weights = HashEmbedding(num_embeddings=num_embeddings, num_hashes=num_hashes,
                                                  embedding_dim=embedding_dim*n_ary_out - num_hashes if append_weight else embedding_dim*n_ary_out,
                                                  num_buckets=num_buckets, append_weight=append_weight)

    def forward(self, input_sequence: torch.Tensor, t_representation: torch.Tensor):
        # input_sequence: (n_batch, n_digits_so_far) input_sequence[b,d] \in {0,n_ary_in}
        # t_representation: (n_batch, n_digits_so_far, n_dim)

        n_digits_so_far = min(self._n_digits, input_sequence.shape[-1])

        # input_sequence_prefix_hashes: (n_batch, n_digits_so_far)
        input_sequence_prefix_hashes = self.transform_sequence_to_prefix_indices(input_sequence)
        # t_weight_: (n_batch, n_digits_so_far, n_ary_out * n_dim)
        t_weight_ = self._logit_layer_weights.forward(input_sequence_prefix_hashes)

        if self._additive:
            # moving average from MSD to d-th digits.
            t_weight_ = torch.cumsum(t_weight_, dim=1)
            # by dividing number of digits, it may avoid nan error.
            # t_denom: (1, n_digits_so_far, 1)
            t_denom = torch.arange(start=1, end=n_digits_so_far+1, device=t_weight_.device).view(1, -1, 1)
            t_weight_ = t_weight_ / t_denom

        # t_weight: (n_batch, n_digits_so_far, n_ary_out, n_dim)
        t_weight = t_weight_.view((-1, n_digits_so_far, self._n_ary, self._n_dim_emb))
        # t_logits: (n_batch, n_digits_so_far, n_ary_out)
        t_logits = torch.matmul(t_weight, t_representation.unsqueeze(-1)).squeeze(-1)

        if self._logit_adjustment:
            t_logits = super().apply_logit_adjustment(logits=t_logits, sequences=input_sequence)

        return t_logits

    def init_weights(self, *args, **kwargs):
        self._logit_layer_weights.reset_parameters(std=0.00001)

    def summary(self):
        ret = super().summary()
        ret["additive"] = self._additive
        return ret


class AdditiveCodeAwareLogits(torch.nn.Module):

    def __init__(self, n_digits: int, n_ary_in: int, n_ary_out: int, n_dim_emb: int,
                 bias: bool = False,
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

        device = input_sequence.device
        n_digits_so_far = min(self._n_digits, input_sequence.shape[-1])
        lst_base_weights = [self.base_weight_layers[digit](input_sequence[:,digit]) for digit in range(n_digits_so_far)]
        # t_base_weight: (n_batch, n_digits_so_far, n_ary_out * n_dim)
        t_base_weight = torch.stack(lst_base_weights, dim=1)
        if self._depends_on_previous_digits is None:
            t_weight_ = torch.cumsum(t_base_weight, dim=1)
            # by dividing number of digits, it may avoid nan error.
            # t_denom: (1, n_digits_so_far, 1)
            t_denom = torch.arange(start=1, end=n_digits_so_far+1, device=device).view(1, -1, 1)
            t_weight_ = t_weight_ / t_denom
        else:
            t_weight_ = self._ragged_cumsum(t_base_weight, dim=1, stride=min(self._depends_on_previous_digits, n_digits_so_far))
        if self._bias:
            t_weight_ = t_weight_ + self.bias_weights[:n_digits_so_far, :]
        # t_weight: (n_batch, n_digits_so_far, n_ary_out, n_dim)
        t_weight = t_weight_.view((-1, n_digits_so_far, self._n_ary_out, self._n_dim_emb))
        # t_logits: (n_batch, n_digits_so_far, n_ary_out)
        t_logits = torch.matmul(t_weight, t_representation.unsqueeze(-1)).squeeze(-1)

        return t_logits

    def summary(self):
        ret = {}
        for attr_name in ("bias", "depends_on_previous_digits", "n_ary_in", "n_ary_out"):
            ret[attr_name] = getattr(self, f"_{attr_name}")
        return ret


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

    def summary(self):
        ret = {
            "n_seq_len": self.n_seq_len
        }