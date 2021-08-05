#!/usr/bin/env python
# -*- coding:utf-8 -*-

from typing import Dict, Callable, Optional, List, Union, Tuple
from transformers import AutoTokenizer, AutoModel
from transformers.tokenization_utils_base import BatchEncoding
import torch
from collections import defaultdict
import os, sys, io, json
import numpy as np


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
            * `entity_spans`: List[List[List[Tuple(int, int)]]]. sequenceに含まれるentityに対応するsubword span．
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

        if self._return_numpy_array:
            embeddings = embeddings.data.numpy()

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
            "entity_spans": lst_lst_entity_subword_spans
        }
        return dict_ret

    def batch(self, lst_sentence_objects: List[Dict[str, Union[List[str], int, str]]]):
        lst_lst_words = [sentence_obj["words"] for sentence_obj in lst_sentence_objects]
        lst_monosemous_entities = [sentence_obj["monosemous_entities"] for sentence_obj in lst_sentence_objects]

    def _batch_to_sentences(self, lst_sentence_objects: List[Dict[str, Union[List[str], int, str]]]):
        pass



