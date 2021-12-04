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

def sequence_to_str(sequence: List[int], delim: str = "-"):
    return delim.join(map(str, sequence))

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

def pad_and_stack_list_of_tensors(lst_embeddings: List[torch.Tensor], max_sequence_length: Optional[int] = None,
                                  return_sequence_length: bool = False):
    """
    it takes the list of embeddings as the input, then applies zero-padding and stacking to transform it as

    @param lst_embeddings:
    @param max_sequence_length:
    """
    dim = -2 # second last axis. it must be the sequence dimension.

    lst_seq_len = [embeddings.shape[dim] for embeddings in lst_embeddings]
    if max_sequence_length is None:
        max_sequence_length = max(lst_seq_len)
    else:
        n_max = max(lst_seq_len)
        assert max_sequence_length >= n_max, \
            f"`max_sequence_length` must be greater or equal to max. embeddings size: {n_max} > {max_sequence_length}"

    lst_padded_embeddings = [pad_trailing_tensors(e_t, max_sequence_length) for e_t in lst_embeddings]
    stacked_embeddings = torch.stack(lst_padded_embeddings)

    if return_sequence_length:
        return stacked_embeddings, lst_seq_len
    else:
        return stacked_embeddings

def create_multiheadattention_attn_mask(
        query_sequence_length: int, key_value_sequence_length: int,
        target_sequence_length: int, source_sequence_length: int,
        num_heads: int, device: Optional[torch.device] = None):
    params = {
        "size": (num_heads, target_sequence_length, source_sequence_length),
        "fill_value": True,
        "dtype": torch.bool
    }
    if device is not None:
        params["device"] = device

    attn_mask = torch.full(**params)
    attn_mask[:, :query_sequence_length, :key_value_sequence_length] = False
    return attn_mask

def create_multiheadattention_attn_mask_batch(
        lst_query_sequence_lengths: List[int], lst_key_value_sequence_lengths: List[int],
        target_sequence_length: int, source_sequence_length: int,
        num_heads: int, device: Optional[torch.device] = None):

    def _apply_attn_mask(n_query, n_key_value):
        return create_multiheadattention_attn_mask(query_sequence_length=n_query, key_value_sequence_length=n_key_value,
                                                   target_sequence_length=target_sequence_length, source_sequence_length=source_sequence_length,
                                                   num_heads=num_heads, device=device)

    lst_attn_masks = [_apply_attn_mask(n_q, n_kv) for n_q, n_kv in zip(lst_query_sequence_lengths, lst_key_value_sequence_lengths)]
    attn_masks = torch.cat(lst_attn_masks, dim=0)

    return attn_masks

def create_sequence_mask(lst_sequence_lengths: List[int], max_sequence_length: Optional[int] = None,
                         device: Optional[torch.device] = None):
    max_sequence_length = max(lst_sequence_lengths) if max_sequence_length is None else max_sequence_length
    n_seq = len(lst_sequence_lengths)

    params = {
        "size": (n_seq, max_sequence_length),
        "fill_value": True,
        "dtype": torch.bool
    }
    if device is not None:
        params["device"] = device

    seq_mask = torch.full(**params)
    for seq_idx, seq_len in enumerate(lst_sequence_lengths):
        seq_mask[seq_idx, :seq_len] = False

    return seq_mask