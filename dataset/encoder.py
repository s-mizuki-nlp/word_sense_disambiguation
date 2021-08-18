#!/usr/bin/env python
# -*- coding:utf-8 -*-

from typing import Dict, Callable, Optional, List, Union, Tuple, Iterable
from transformers import AutoTokenizer, AutoModel
from transformers.tokenization_utils_base import BatchEncoding
import torch
import numpy as np

from .utils import preprocessor_for_monosemous_entity_annotated_corpus
from .utils import numpy_to_tensor, get_dtype_and_device, Array_like

class BERTEmbeddings(object):

    def __init__(self, model_or_name: Union[str, AutoModel] = None, tokenizer: Optional[AutoTokenizer] = None,
                 layers: List[int] = [-4,-3,-2,-1],
                 return_compressed_format: bool = True,
                 return_numpy_array: bool = False,
                 **kwargs):
        if isinstance(model_or_name, str):
            self._tokenizer = AutoTokenizer.from_pretrained(model_or_name)

            kwargs["output_hidden_states"] = True
            self._model = AutoModel.from_pretrained(model_or_name, **kwargs)
        else:
            assert tokenizer is not None, f"you must specify `tokenizer` instance."
            self._model = model_or_name
            self._tokenizer = tokenizer
        self._layers = layers
        self._hidden_size = self._model.config.hidden_size
        self._return_compressed_format = return_compressed_format
        self._return_numpy_array = return_numpy_array

    def __call__(self, lst_lst_words: Union[List[List[str]], List[str]],
                 lst_lst_entity_spans: Union[List[List[Tuple[int, int]]], List[Tuple[int, int]]],
                 add_special_tokens: bool = True,
                 **kwargs):
        """

        @param lst_lst_words: list of the sequence of words.
        @param lst_lst_entity_spans: list of the list of entity spans.
        @param add_special_tokens: whether append [CLS] and [SEP] special tokens or not. DEFAULT: True
        @param kwargs: keyword arguments that are passed to tokenizer.batch_encode_plus() method.
        @return: Dict[str, Any]. embeddings, sequence information, and entity span information.
            * `embeddings`: subword embeddings. $x^{l}_{t}$をポジションtのsubwordに対する$l$層目の内部表現として $\sum_{l \in \mathrm{layers}}{x^{l}_{t}}$ を出力．
                shape = (sum(sequence_lengths), hidden_size) if return_compressed_format=True else (batch_size, max(sequence_lengths), hidden_size)
            * `sequence_lengths`: shape (batch_size,). subword sequenceの長さ．
            * `sequence_spans`: shape (batch_size, 2). subword sequenceに対応するspan．return_compressed_format=Trueの場合のみ出力．
            * `entity_spans`: List[List[List[Tuple[int, int]]]]. sequenceに含まれるentityに対応するsubword span．
                ひとつのspanが1単語に対応する．複合語entityの場合は複数のspanが返される．
        """
        is_batch_input = True
        if isinstance(lst_lst_words[0], str):
            lst_lst_words = [lst_lst_words]
        if isinstance(lst_lst_entity_spans[0][0], int):
            lst_lst_entity_spans = [lst_lst_entity_spans]
        assert len(lst_lst_words) == len(lst_lst_entity_spans), f"batch size mismatch detected."
        if len(lst_lst_words) == 1:
            is_batch_input = False

        # encode word sequences
        kwargs["padding"] = True
        kwargs["return_attention_mask"] = True
        kwargs["return_tensors"] = "pt"
        token_info: BatchEncoding = self._tokenizer.batch_encode_plus(lst_lst_words, is_split_into_words=True,
                                                       add_special_tokens=add_special_tokens,
                                                       **kwargs)

        # calculate subword-level entity spans
        lst_lst_entity_subword_spans = []
        # loop within batch
        for batch_idx, lst_entity_span in enumerate(lst_lst_entity_spans):
            # loop within entities
            entity_subword_spans = []
            for entity_span in lst_entity_span:
                subword_spans = []
                for word_index in range(entity_span[0], entity_span[1]):
                    span_info = token_info.word_to_tokens(batch_or_word_index=batch_idx, word_index=word_index)
                    tup_span = (span_info.start, span_info.end)
                    subword_spans.append(tup_span)
                entity_subword_spans.append(subword_spans)
            lst_lst_entity_subword_spans.append(entity_subword_spans)

        # calculate sequence length
        v_seq_length = token_info["attention_mask"].sum(axis=-1).data.numpy()
        if self._return_compressed_format:
            v_temp = np.cumsum(np.concatenate(([0], v_seq_length)))
            v_seq_spans = np.stack([[s, t] for s, t in zip(v_temp[:-1], v_temp[1:])])
        else:
            v_seq_spans = None

        # encode to embeddings
        with torch.no_grad():
            obj_encoded = self._model.forward(**token_info)
        embeddings = torch.zeros_like(obj_encoded.last_hidden_state)
        it_hidden_states = (obj_encoded.hidden_states[i] for i in self._layers)
        for hidden_state in it_hidden_states:
            embeddings += hidden_state

        if self._return_compressed_format:
            # tensor elements where attention_mask == 1 are valid subword token embeddings.
            mask = token_info["attention_mask"].unsqueeze(-1) == 1
            # last dimension (along hidden_size axis) will be broadcasted.
            # output: (sum(v_seq_length), hidden_size)
            embeddings = torch.masked_select(embeddings, mask).reshape(-1, self._hidden_size)
            assert embeddings.shape[0] == v_seq_length.sum(), f"sequence size mismatch detected."

        attention_mask = token_info["attention_mask"]

        if self._return_numpy_array:
            embeddings = embeddings.data.numpy()
            attention_mask = attention_mask.data.numpy()

        # return first element if input is not batch.
        if is_batch_input == False:
            lst_lst_entity_subword_spans = lst_lst_entity_subword_spans[0]
            v_seq_length = v_seq_length[0]
            v_seq_spans = v_seq_spans[0] if v_seq_spans is not None else None
            if not self._return_compressed_format:
                embeddings = embeddings[0]

        dict_ret = {
            "embeddings": embeddings,
            "sequence_lengths": v_seq_length,
            "sequence_spans": v_seq_spans,
            "entity_spans": lst_lst_entity_subword_spans,
            "attention_mask": attention_mask
        }
        return dict_ret


def convert_compressed_format_to_batch_format(embeddings: Array_like,
                                              lst_sequence_lengths: Iterable[int],
                                              max_sequence_length: Optional[int] = None,
                                              return_tensor: bool = False):
    """
    converts compressed format (\sum{seq_len}, n_dim) into standard format (n_seq, max_seq_len, n_dim)

    @param embeddings: embeddings with compressed format.
    @param lst_sequence_lengths: list of sequence length.
    @param max_sequence_length: max. sequence length.
    """
    if torch.is_tensor(embeddings):
        embeddings = embeddings.data.numpy()

    if max_sequence_length is None:
        max_sequence_length = max(lst_sequence_lengths)

    n_seq = len(lst_sequence_lengths)
    n_seq_len_sum, n_dim = embeddings.shape
    assert sum(lst_sequence_lengths) == n_seq_len_sum, \
        f"total sequence length doesn't match with compressed embeddings dimension size: ({n_seq_len_sum}, {n_dim})"

    # embeddings: (n_seq, max_seq_len, n_dim)
    shape = (n_seq, max_sequence_length, n_dim)
    dtype = embeddings.dtype
    embeddings_decompressed = np.zeros(shape, dtype)

    v_temp = np.cumsum(np.concatenate(([0], lst_sequence_lengths)))
    lst_seq_spans = [slice(s,t) for s, t in zip(v_temp[:-1], v_temp[1:])]

    for idx, span in enumerate(lst_seq_spans):
        seq_len = span.stop - span.start
        embeddings_decompressed[idx, :seq_len, :] = embeddings[span, :]

    # attention_mask: (n_seq, max_seq_len)
    shape = (n_seq, max_sequence_length)
    attention_mask = np.zeros(shape, np.int64)
    for idx, seq_len in enumerate(lst_sequence_lengths):
        attention_mask[idx, :seq_len] = 1

    if return_tensor:
        embeddings_decompressed = numpy_to_tensor(embeddings_decompressed)
        attention_mask = numpy_to_tensor(attention_mask)

    dict_ret = {
        "embeddings": embeddings_decompressed,
        "attention_mask": attention_mask
    }
    return dict_ret


def calc_entity_embeddings_from_subword_embeddings(subword_embeddings: torch.Tensor,
                                                   lst_lst_lst_entity_subword_spans: List[List[List[Tuple[int, int]]]],
                                                   return_compressed_format: bool = False,
                                                   max_entity_size: Optional[int] = None):
    assert subword_embeddings.ndim == 3, f"embeddings dimension must be (n_batch, max_seq_len, n_dim)."
    assert subword_embeddings.shape[0] == len(lst_lst_lst_entity_subword_spans), f"batch size must be identical to entity subword spans."

    lst_num_entities = list(map(len, lst_lst_lst_entity_subword_spans))
    n_entities_sum = sum(lst_num_entities)
    max_entity_size = max(lst_num_entities) if max_entity_size is None else max_entity_size

    # entity_embeddings: (\sum(n_entities), n_dim)
    n_seq, max_seq_len, n_dim = subword_embeddings.shape
    shape = (n_entities_sum, n_dim)
    dtype, device = get_dtype_and_device(subword_embeddings)
    entity_embeddings = torch.zeros(shape, dtype=dtype, device=device)

    entity_cursor = 0
    for seq_idx, lst_lst_entity_subword_spans in enumerate(lst_lst_lst_entity_subword_spans):
        for entity_idx, lst_entity_subword_spans in enumerate(lst_lst_entity_subword_spans):
            entity_tensor = torch.zeros(n_dim, dtype=dtype, device=device)
            n_words = len(lst_entity_subword_spans)
            for word_idx, entity_subword_spans in enumerate(lst_entity_subword_spans):
                word_tensor = subword_embeddings[seq_idx, slice(*entity_subword_spans), :].mean(axis=0)
                entity_tensor = entity_tensor + word_tensor
            if n_words > 1:
                entity_tensor = entity_tensor / n_words
            entity_embeddings[entity_cursor, :] = entity_tensor

            entity_cursor += 1

    if not return_compressed_format:
        dict_ret = convert_compressed_format_to_batch_format(embeddings=entity_embeddings,
                                                             lst_sequence_lengths=lst_num_entities,
                                                             max_sequence_length=max_entity_size,
                                                             return_tensor=True
                                                             )
        dict_ret["num_entities"] = lst_num_entities
    else:
        dict_ret = {
            "embeddings": subword_embeddings,
            "num_entities": lst_num_entities,
            "attention_mask": None
        }

    return dict_ret