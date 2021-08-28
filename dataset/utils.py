#!/usr/bin/env python
# -*- coding:utf-8 -*-

from typing import Dict, Callable, Optional, List, Union, Tuple, Any
import numpy as np
import torch

Array_like = Union[torch.Tensor, np.ndarray]


def preprocessor_for_monosemous_entity_annotated_corpus(record: Dict[str, Any]):
    lst_words = record["words"]
    lst_entity_spans = [entity["span"] for entity in record["monosemous_entities"]]
    return lst_words, lst_entity_spans


def numpy_to_tensor(object: Array_like) -> torch.Tensor:
        if isinstance(object, torch.Tensor):
            return object
        elif isinstance(object, np.ndarray):
            return torch.from_numpy(object)
        else:
            raise TypeError(f"unsupported type: {type(object)}")

def tensor_to_numpy(object: Array_like) -> torch.Tensor:
        if isinstance(object, torch.Tensor):
            return object.cpu().data.numpy()
        elif isinstance(object, np.ndarray):
            return object
        else:
            raise TypeError(f"unsupported type: {type(object)}")

def get_dtype_and_device(t: torch.Tensor):
    return t.dtype, t.device

def lemma_pos_to_tuple(lemma: str, pos: str, lemma_lowercase: bool, **kwargs):
    if lemma_lowercase:
        return (lemma.lower(), pos)
    else:
        return (lemma, pos)

def pad_trailing_tensors(embeddings: torch.Tensor, n_length_after_padding: int):
    """
    pad fixed-valued tensor at the training (=bottom in 2D) of a given tensor to obtain desired vector sequence length.

    @param embeddings: 2D or 3D tensor. shape: ([n_batch], n_sequence, n_dim)
    @param n_length_after_padding: sequence length (=height in 2D) after padding.
    """
    dim = -2 # second last axis. it must be the sequence dimension.

    n_length = embeddings.shape[dim]
    if n_length_after_padding < n_length:
        raise ValueError(f"`n_length_after_padding` must be longer than current length: {n_length} < {n_length_after_padding}")

    n_pad = n_length_after_padding - n_length
    padding_function = torch.nn.ZeroPad2d(padding=(0,0,0,n_pad))
    embeddings_padded = padding_function(embeddings)

    return embeddings_padded
