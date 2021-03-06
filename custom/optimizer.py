#!/usr/bin/env python
# -*- coding:utf-8 -*-
import math

from torch.optim import Adam
from torch.optim.lr_scheduler import _LRScheduler

class AdamWithWarmup(_LRScheduler):
    """
    Optim wrapper that implements rate.
    ref: The Annotated Transformer. https://nlp.seas.harvard.edu/2018/04/03/attention.html#optimizer
    """
    def __init__(self, model_size, factor, warmup, optimizer, **kwargs):
        super().__init__(optimizer=optimizer)
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def lr_max(self):
        return self.factor / math.sqrt(self.warmup * self.model_size)


def init_adam_with_warmup_optimizer(model, n_dim_hidden, factor:int = 2, warmup: int=4000, beta1=0.9, beta2=0.9995, eps=1e-9):
    opt = Adam(model.parameters(), lr=0, betas=(beta1, beta2), eps=eps)
    return AdamWithWarmup(n_dim_hidden, factor, warmup, opt)