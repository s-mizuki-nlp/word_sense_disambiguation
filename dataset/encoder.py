#!/usr/bin/env python
# -*- coding:utf-8 -*-
import warnings
from typing import Dict, Callable, Optional, List, Union, Tuple, Iterable
from transformers import AutoTokenizer, AutoModel
from transformers.tokenization_utils_base import BatchEncoding
import torch
import numpy as np

from .utils import pad_trailing_tensors
from .utils import numpy_to_tensor, tensor_to_numpy, get_dtype_and_device, Array_like

class BERTEmbeddings(object):

    def __init__(self, model_or_name: Union[str, AutoModel] = None, tokenizer: Optional[AutoTokenizer] = None,
                 layers: List[int] = [-4,-3,-2,-1],
                 return_compressed_format: bool = True,
                 return_numpy_array: bool = False,
                 ignore_too_long_sequence: bool = True,
                 device_ids: Optional[List[int]] = None,
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
        self._ignore_too_long_sequence = ignore_too_long_sequence

        self._device_ids = device_ids
        if isinstance(device_ids, list):
            if len(device_ids) == 1:
                self._device = torch.device(f"cuda:{device_ids[0]}")
                self._model = self._model.to(self._device)
            else:
                self._device = torch.device(f"cuda:{device_ids[0]}")
                self._model = torch.nn.DataParallel(self._model, device_ids=device_ids, output_device=device_ids[-1]).to(self._device)
        else:
            self._device = torch.device("cpu")

    def weighted_average_masked_word_embeddings(self,
                                                mat_embeddings_orig: np.ndarray,
                                                mat_embeddings_masked: np.ndarray,
                                                lst_lst_entity_spans_orig: List[List[Tuple[int,int]]],
                                                lst_lst_entity_spans_masked: List[List[Tuple[int,int]]],
                                                alpha: float
                                                ) -> np.ndarray:
        mat_mwe = mat_embeddings_orig.copy()
        # iterate over entities
        for lst_entity_spans, lst_masked_entity_spans in zip(lst_lst_entity_spans_orig, lst_lst_entity_spans_masked):
            # iterate over words in entity:
            for entity_word_span, masked_entity_word_span in zip(lst_entity_spans, lst_masked_entity_spans):
                # mat_subword_embeddings: (n_subwords, n_dim)
                mat_subword_embeddings = mat_embeddings_orig[slice(*entity_word_span),:]
                # vec_masked_word_embedding: (1, n_dim)
                vec_masked_word_embedding = mat_embeddings_masked[slice(*masked_entity_word_span),:]
                assert vec_masked_word_embedding.shape[0] == 1, f"[MASK] seems to be split into subwords?"

                # entity subword embeddings will be averaged over original embeddings and masked embedding.
                mat_mwe[slice(*entity_word_span)] = (1.0 - alpha) * mat_subword_embeddings + alpha * vec_masked_word_embedding

        return mat_mwe

    def batch_weighted_average_masked_word_embeddings(self,
                                                mat_embeddings_orig: np.ndarray,
                                                mat_embeddings_masked: np.ndarray,
                                                lst_sequence_spans_orig: np.ndarray,
                                                lst_sequence_spans_masked: np.ndarray,
                                                lst_lst_lst_entity_spans_orig: List[List[List[Tuple[int,int]]]],
                                                lst_lst_lst_entity_spans_masked: List[List[List[Tuple[int,int]]]],
                                                alpha: float
                                                ) -> np.ndarray:
        """
        calculate weighted average over entity subword embeddings between original embeddings and masked word embeddings.

        @param mat_embeddings_orig: stacked subword embeddings of the sentences in mini-batch.
        @param mat_embeddings_masked: subword embeddings but applied masking.
        @param lst_sequence_spans_orig: list of sequence spans of subword embedding for each sentence.
        @param lst_sequence_spans_masked: sequence spans but applied masking.
        @param lst_lst_lst_entity_spans_orig: subword spans of the words in the entities.
        @param lst_lst_lst_entity_spans_masked: subword spans but applied masking.
        @param alpha: averaging weight. 0.0 = use original embedding, 1.0 = use masked embedding.
        @return: weighted average between original embeddings and masked word embeddings.
        """
        mat_mwe = mat_embeddings_orig.copy()
        # iterate over sentences
        it_record_pairs = zip(lst_sequence_spans_orig, lst_sequence_spans_masked, lst_lst_lst_entity_spans_orig, lst_lst_lst_entity_spans_masked)
        for seq_span_orig, seq_span_masked, lst_lst_entity_spans_orig, lst_lst_entity_spans_masked in it_record_pairs:
            mat_embeddings_orig_s = mat_embeddings_orig[slice(*seq_span_orig),:]
            mat_embeddings_masked_s = mat_embeddings_masked[slice(*seq_span_masked),:]
            mat_mwe_s = self.weighted_average_masked_word_embeddings(
                mat_embeddings_orig=mat_embeddings_orig_s,
                mat_embeddings_masked=mat_embeddings_masked_s,
                lst_lst_entity_spans_orig=lst_lst_entity_spans_orig,
                lst_lst_entity_spans_masked=lst_lst_entity_spans_masked,
                alpha=alpha
            )
            mat_mwe[slice(*seq_span_orig),:] = mat_mwe_s

        return mat_mwe

    def tokenize(self, lst_lst_words, add_special_tokens: bool, **kwargs) -> BatchEncoding:

        # encode word sequences
        kwargs["padding"] = True
        kwargs["return_attention_mask"] = True
        kwargs["return_tensors"] = "pt"
        token_info = self._tokenizer.batch_encode_plus(lst_lst_words, is_split_into_words=True,
                                                       add_special_tokens=add_special_tokens,
                                                       **kwargs)
        _ = token_info.to(self._device)

        return token_info

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

        # tokenize
        token_info = self.tokenize(lst_lst_words=lst_lst_words, add_special_tokens=add_special_tokens, **kwargs)

        # verify
        if self._ignore_too_long_sequence:
            n_seq_len = token_info["input_ids"].shape[-1]
            if n_seq_len > self._tokenizer.model_max_length:
                warnings.warn("sequence is too long. skip encoding.")
                return {}

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
        v_seq_length = tensor_to_numpy(token_info["attention_mask"].sum(axis=-1))
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
            embeddings = tensor_to_numpy(embeddings)
            attention_mask = tensor_to_numpy(attention_mask)

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

    @property
    def tokenizer(self):
        return self._tokenizer


def convert_compressed_format_to_list_of_tensors(embeddings: Array_like,
                                                 lst_sequence_lengths: Iterable[int],
                                                 padding: bool = False,
                                                 max_sequence_length: Optional[int] = None
                                                 ) -> List[torch.Tensor]:
    """
    converts compressed format (\sum{seq_len}, n_dim) into the list of tensors [(n_len_0, n_dim), (n_len_1, n_dim), ...]

    @param embeddings: embeddings with compressed format.
    @param lst_sequence_lengths: list of sequence length.
    @param padding: pad with zero-valued tensor to maximum sequence length or not.
    @param max_sequence_length: explicit max. sequence length. DEFAULT: max(lst_sequence_lengths)
    """
    embeddings = numpy_to_tensor(embeddings)

    if max_sequence_length is None:
        max_sequence_length = max(lst_sequence_lengths)

    n_seq_len_sum, n_dim = embeddings.shape
    assert sum(lst_sequence_lengths) == n_seq_len_sum, \
        f"total sequence length doesn't match with compressed embeddings dimension size: ({n_seq_len_sum}, {n_dim})"

    # lst_embeddings: [(max_seq_len, n_dim)]*n_seq
    v_temp = np.cumsum(np.concatenate(([0], lst_sequence_lengths)))
    lst_seq_spans = [slice(s,t) for s, t in zip(v_temp[:-1], v_temp[1:])]

    # extract spans from compressed-view embedding.
    lst_embeddings = [embeddings[span,:] for span in lst_seq_spans]

    # pad tensors if we need
    if padding:
        lst_embeddings = list(map(lambda emb_t: pad_trailing_tensors(emb_t, max_sequence_length), lst_embeddings))

    return lst_embeddings

def convert_compressed_format_to_batch_format(embeddings: Array_like,
                                              lst_sequence_lengths: Iterable[int],
                                              max_sequence_length: Optional[int] = None
                                              ) -> Dict[str, torch.Tensor]:
    """
    converts compressed format (\sum{seq_len}, n_dim) into standard format (n_seq, max_seq_len, n_dim)

    @param embeddings: embeddings with compressed format.
    @param lst_sequence_lengths: list of sequence length.
    @param max_sequence_length: max. sequence length.
    """
    n_seq = len(lst_sequence_lengths)
    if max_sequence_length is None:
        max_sequence_length = max(lst_sequence_lengths)

    # compressed -> list of padded format -> stack
    lst_padded_embeddings = convert_compressed_format_to_list_of_tensors(embeddings, lst_sequence_lengths, padding=True,
                                                                         max_sequence_length=max_sequence_length)
    embeddings_decompressed = torch.stack(lst_padded_embeddings, dim=0)

    # attention_mask: (n_seq, max_seq_len)
    shape = (n_seq, max_sequence_length)
    attention_mask = np.zeros(shape, np.int64)
    for idx, seq_len in enumerate(lst_sequence_lengths):
        attention_mask[idx, :seq_len] = 1
    attention_mask = numpy_to_tensor(attention_mask)

    dict_ret = {
        "embeddings": embeddings_decompressed,
        "attention_mask": attention_mask
    }
    return dict_ret

def calc_entity_subwords_average_vectors(context_embeddings: Array_like,
                                         lst_lst_entity_subword_spans: List[List[Tuple[int, int]]]
                                         ) -> List[torch.Tensor]:
    context_embeddings = numpy_to_tensor(context_embeddings)
    assert context_embeddings.ndim == 2, f"embeddings dimension must be (n_seq_len, n_dim)."

    # entity_embeddings: List[torch.tensor(n_window, n_dim)]
    lst_entity_vectors = []
    for entity_idx, lst_entity_subword_spans in enumerate(lst_lst_entity_subword_spans):
        # take average at word level, then take average at entity level.
        lst_word_vectors = [context_embeddings[slice(*subword_span), :].mean(dim=0) for subword_span in lst_entity_subword_spans]
        entity_vector = torch.stack(lst_word_vectors).mean(dim=0)
        lst_entity_vectors.append(entity_vector)

    return lst_entity_vectors

def extract_entity_subword_embeddings(context_embeddings: Array_like,
                                      lst_lst_entity_subword_spans: List[List[Tuple[int, int]]],
                                      padding: bool = False,
                                      max_entity_subword_length: Optional[int] = None) -> Dict[str, Union[Array_like, List[int]]]:
    """
    extracts entity spans from context embeddings and returns as the list of tensors.

    @param context_embeddings: sequence of subword embeddings of a sentence. shape: (n_subwords, n_dim)
    @param lst_lst_entity_subword_spans: list of the list of entity subword spans in a sentence.
    @param padding: pad entity subword span or not. DEFAULT: False
    @param max_entity_subword_length: max. width of the entity span.
    @return: dictionary. embeddings: List[(n_window, n_dim)], sequence_lengths: List[n_window]
    """
    is_input_tensor = torch.is_tensor(context_embeddings)
    context_embeddings = numpy_to_tensor(context_embeddings)

    assert context_embeddings.ndim == 2, f"`context_embeddings` dimension must be (max_seq_len, n_dim)."

    # extract entity subword embeddings as List[(n_subwords, n_dim)]
    lst_embeddings = []
    for lst_entity_subword_spans in lst_lst_entity_subword_spans:
        start = lst_entity_subword_spans[0][0]
        stop = lst_entity_subword_spans[-1][1]
        entity_subword_embeddings = context_embeddings[start:stop, :]
        lst_embeddings.append(entity_subword_embeddings)

    # calculate sequence length
    lst_sequence_lengths = [embeddings.shape[0] for embeddings in lst_embeddings]

    # apply padding
    if padding:
        n_window_max = max(lst_sequence_lengths)
        max_entity_subword_length = n_window_max if max_entity_subword_length is None else max_entity_subword_length
        lst_embeddings = list(map(lambda emb_t: pad_trailing_tensors(emb_t, max_entity_subword_length), lst_embeddings))

    if not is_input_tensor:
        lst_embeddings = list(map(tensor_to_numpy, lst_embeddings))

    # store them
    dict_ret = {
        "embeddings": lst_embeddings,
        "sequence_lengths": lst_sequence_lengths
    }

    return dict_ret
