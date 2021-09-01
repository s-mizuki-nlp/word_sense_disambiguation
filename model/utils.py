#!/usr/bin/env python
# -*- coding:utf-8 -*-
import torch


def masked_average(embeddings: torch.Tensor, sequence_mask: torch.BoolTensor, dim: int = -2):
    """
    apply masked average.

    @param embeddings: 2D or 3D embeddings with second last axes is the sequence dimension.
            shape: (n_batch, n_seq, n_dim) or (n_seq, n_dim)
    @param sequence_mask: 1D or 2D tensor where invalid sequence element value is True.
            shape: (n_batch, n_seq) or (n_seq,)
    @param dim: sequence dimension. DEFAULT: -2
    @rtype: averaged embedding along with sequence dimension. shape: (n_batch, n_dim) or (n_dim,)
    """
    assert embeddings.ndim == sequence_mask.ndim + 1, f"embeddings and mask dimension mismatch."

    # (n_batch, n_seq) -> (n_batch, n_seq, 1)
    mask_rev = ~sequence_mask.unsqueeze(dim=-1)
    # (n_batch, n_dim) / (n_batch, 1)
    t_mean = (embeddings * mask_rev).nansum(dim=dim) / (mask_rev.sum(dim=dim))

    return t_mean