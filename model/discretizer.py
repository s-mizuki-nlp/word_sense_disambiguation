#!/usr/bin/env python
# -*- coding:utf-8 -*-

from typing import List, Optional
import torch
from torch import nn
from torch.nn import functional as F
from entmax.activations import entmax15, sparsemax
import numpy as np

_EPS = 1E-6

class StraightThroughEstimator(nn.Module):

    def __init__(self, add_gumbel_noise: bool = False, temperature: float = 1.0):
        super(StraightThroughEstimator, self).__init__()
        self._add_gumbel_noise = add_gumbel_noise
        self._temperature = temperature

    def forward(self, probs, dim: int = -1):

        if self._add_gumbel_noise:
            logits = torch.log(probs+_EPS)
            output = F.gumbel_softmax(logits=logits, tau=self._temperature, hard=True, dim=dim)
            return output

        # one-hot operation
        t_z = probs
        index = t_z.max(dim=dim, keepdim=True)[1]
        y_hard = torch.zeros_like(t_z).scatter_(dim, index, 1.0)
        output = y_hard - t_z.detach() + t_z

        return output

    @property
    def temperature(self):
        return self._temperature

    @temperature.setter
    def temperature(self, value):
        self._temperature = value


class GumbelSoftmax(nn.Module):

    def __init__(self, temperature: float = 1.0, add_gumbel_noise: bool = True, **kwargs):
        super(GumbelSoftmax, self).__init__()
        self._add_gumbel_noise = add_gumbel_noise
        self._temperature = temperature

        if not add_gumbel_noise:
            Warning("it reduces to a simple softmax activation.")

    def forward(self, probs, dim: int = -1):

        logits = torch.log(probs+_EPS)

        if self._add_gumbel_noise:
            return F.gumbel_softmax(logits=logits, tau=self._temperature, hard=False, dim=dim)
        else:
            return F.softmax(input=logits / self._temperature, dim=dim)

    @property
    def temperature(self):
        return self._temperature

    @temperature.setter
    def temperature(self, value):
        self._temperature = value

    def summary(self):
        ret = {
            "class_name": self.__class__.__name__,
            "temperature": self.temperature
        }
        return ret


class Entmax15Estimator(nn.Module):

    def __init__(self, add_gumbel_noise: bool = False, temperature: float = 1.0):
        super(Entmax15Estimator, self).__init__()
        self._add_gumbel_noise = add_gumbel_noise
        self._temperature = temperature

    def _gumbel_noise(self, logits):
        gumbels = -torch.empty_like(logits).exponential_().log()  # ~Gumbel(0,1)
        gumbels = (logits + gumbels) / self._temperature  # ~Gumbel(logits,tau)

        return gumbels

    def forward(self, probs, dim: int = -1):
        logits = torch.log(probs+_EPS)
        if self._add_gumbel_noise:
            logits = self._gumbel_noise(logits=logits)

        return entmax15(logits, dim=dim)

    @property
    def temperature(self):
        return self._temperature

    @temperature.setter
    def temperature(self, value):
        self._temperature = value


class SparsemaxEstimator(nn.Module):

    def __init__(self, add_gumbel_noise: bool = False, temperature: float = 1.0):
        super(SparsemaxEstimator, self).__init__()
        self._add_gumbel_noise = add_gumbel_noise
        self._temperature = temperature

    def _gumbel_noise(self, logits):
        gumbels = -torch.empty_like(logits).exponential_().log()  # ~Gumbel(0,1)
        gumbels = (logits + gumbels) / self._temperature  # ~Gumbel(logits,tau)

        return gumbels

    def forward(self, probs, dim: int = -1):
        logits = torch.log(probs+_EPS)
        if self._add_gumbel_noise:
            logits = self._gumbel_noise(logits=logits)

        return sparsemax(logits, dim=dim)

    @property
    def temperature(self):
        return self._temperature

    @temperature.setter
    def temperature(self, value):
        self._temperature = value


class MaskedGumbelSoftmax(GumbelSoftmax):

    def _dtype_and_device(self, t: torch.Tensor):
        return t.dtype, t.device

    def forward(self, probs, dim: int = -1):
        """
        apply gumbel-softmax trick only on nonzero probabilities.

        @param probs: code probability distribution. shape: (n_batch, *, n_ary)
        @param dim: dimension that GS trick is applied along with.
        @return: sampled probability distribution.
        """
        dtype, device = self._dtype_and_device(probs)
        n_ary = probs.shape[-1]

        # split p(C_n=0|x) and p(C_n!=0|x)
        # t_p_c_zero: (n_batch, n_digits, 1)
        probs_zero = torch.index_select(probs, dim=-1, index=torch.tensor(0, device=device))
        # t_p_c_nonzero: (n_batch, n_digits, n_ary-1)
        probs_nonzero = torch.index_select(probs, dim=-1, index=torch.arange(1, n_ary, dtype=torch.long, device=device))

        # apply gumbel-softmax trick only on nonzero probabilities
        gumbels_nonzero = super().forward(probs_nonzero, dim=-1)

        # concat with zero-probs and nonzero-probs
        y_soft = torch.cat((probs_zero, (1.0-probs_zero)*gumbels_nonzero), dim=-1)

        return y_soft

    @property
    def temperature(self):
        return self._temperature

    @temperature.setter
    def temperature(self, value):
        self._temperature = value
