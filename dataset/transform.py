#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

from typing import Dict, Callable, Optional
from collections import defaultdict
import os, sys, io, json
import numpy as np

from dataset_preprocessor import utils_wordnet


class EmbeddingNormalizer(object):

    def __init__(self, field_name_embedding="embedding"):
        self._field_name = field_name_embedding

    def __call__(self, sample):

        x = sample[self._field_name]
        sample[self._field_name] = x / np.linalg.norm(x)

        return sample


class HyponymyEntryToListOfHyponymyPair(object):

    def __init__(self, field_name_hyponym: str = "hyponym", field_name_hypernym: str = "hypernym", field_name_hypernyms: str = "hypernyms"):
        self._field_name_hyponym = field_name_hyponym
        self._field_name_hypernym = field_name_hypernym
        self._field_name_hypernyms = field_name_hypernyms

    def __call__(self, sample):
        hyponym = sample.get(self._field_name_hyponym, None)
        hypernym = sample.get(self._field_name_hypernym, None)
        hypernyms = sample.get(self._field_name_hypernyms, None)

        assert hyponym is not None, "could not get hyponym."

        if hypernym is not None:
            return [(hyponym, hypernym)]
        elif isinstance(hypernyms, list):
            return [(hyponym, hyper) for hyper in hypernyms]
        else:
            raise AssertionError("could not get hypernym.")


class FieldTypeConverter(object):

    def __init__(self, dict_field_type_converter: Dict[str, Callable]):

        self._dict_field_type_converter = dict_field_type_converter

    def __call__(self, sample):

        for field_name, converter in self._dict_field_type_converter.items():
            if field_name in sample:
                sample[field_name] = converter(sample[field_name])

        return sample


class ToWordNetPoSTagConverter(object):

    def __init__(self, pos_field_name: str = "pos",
                 mwe_pos_field_name: str = "mwe_pos"):

        self._pos_field_name = pos_field_name
        self._mwe_pos_field_name = mwe_pos_field_name

    def __call__(self, lst_tokens):

        # Stanford token PoS tags to WordNet PoS tags.
        for token in lst_tokens:
            token[self._pos_field_name] = utils_wordnet.ptb_tagset_to_wordnet_tagset(token[self._pos_field_name])
            if self._mwe_pos_field_name in token:
                token[self._mwe_pos_field_name] = utils_wordnet.universal_tagset_to_wordnet_tagset(token[self._mwe_pos_field_name])

        return lst_tokens

class ToWordNetPoSTagAndLemmaConverter(ToWordNetPoSTagConverter):

    def __init__(self,
                 pos_field_name: str = "pos",
                 mwe_pos_field_name: str = "mwe_pos",
                 lemma_field_name: str = "lemma",
                 mwe_lemma_field_name: str = "mwe_lemma",
                 lemma_and_pos_field_name: str = "lemma_pos",
                 mwe_lemma_and_pos_field_name: str = "mwe_lemma_pos",
                 lowercase: bool = True):

        super().__init__(pos_field_name, mwe_pos_field_name)
        self._lemma_field_name = lemma_field_name
        self._mwe_lemma_field_name = mwe_lemma_field_name
        self._lemma_and_pos_field_name = lemma_and_pos_field_name
        self._mwe_lemma_and_pos_field_name = mwe_lemma_and_pos_field_name
        self._lowercase = lowercase

    def __call__(self, lst_tokens):
        lst_tokens = super().__call__(lst_tokens)
        if self._lowercase:
            return self._create_tuple_lowercase(lst_tokens)
        else:
            return self._create_tuple(lst_tokens)

    def _create_tuple_lowercase(self, lst_tokens):
        for token in lst_tokens:
            token[self._lemma_and_pos_field_name] = (token[self._lemma_field_name].lower(), token[self._pos_field_name])
            if self._mwe_pos_field_name in token:
                token[self._mwe_lemma_and_pos_field_name] = (token[self._mwe_lemma_field_name].lower(), token[self._mwe_pos_field_name])

        return lst_tokens

    def _create_tuple(self, lst_tokens):
        for token in lst_tokens:
            token[self._lemma_and_pos_field_name] = (token[self._lemma_field_name], token[self._pos_field_name])
            if self._mwe_pos_field_name in token:
                token[self._mwe_lemma_and_pos_field_name] = (token[self._mwe_lemma_field_name], token[self._mwe_pos_field_name])

        return lst_tokens


class FrequencyBasedMonosemousEntitySampler(object):

    def __init__(self,
                 min_freq: Optional[int] = None,
                 max_freq: Optional[int] = None,
                 path_monosemous_words_freq: Optional[str] = None,
                 lemma_lowercase: bool = True
                 ):
        """
        単義語をsamplingする．指定頻度未満の単義語は削除，指定頻度以上の単義語はdown-samplingする．

        @param min_freq: 最小頻度．指定値未満の単義語は削除．
        @param max_freq: 最大頻度．指定値を上回る単義語は，先頭max_freq個のみ採択，残りは削除．
        @param path_monosemous_words_freq: 単義語の頻度データ．フォーマットはNDJSON, `{'lemma': 'toddler', 'pos': 'n', 'freq': 2681}`

        @rtype: object
        """

        if isinstance(min_freq, int):
            assert path_monosemous_words_freq is not None, f"you must specify `path_monosemous_words_freq` argument."
            assert os.path.exists(path_monosemous_words_freq), f"specified file does not exist: {path_monosemous_words_freq}"

        self._min_freq = 1 if min_freq is None else min_freq
        self._max_freq = float("inf") if max_freq is None else max_freq
        self._lemma_pos_freq = self._load_lemma_pos_freq(path=path_monosemous_words_freq)
        self._lemma_pos_freq_so_far = defaultdict(int)
        self._lemma_lowercase = lemma_lowercase

    def _load_lemma_pos_freq(self, path: str):
        dict_freq = {}
        ifs = io.open(path, mode="r")
        for record in ifs:
            d = json.loads(record.strip())
            key = (d["lemma"], d["pos"])
            dict_freq[key] = d["freq"]
        ifs.close()

        return dict_freq

    def __call__(self, lst_entities: Dict[str, str]):
        """
        @param lst_entities: dict like `{'lemma': 'Dubonnet', 'pos': 'n', 'occurence': 0, 'span': [1, 2]}`
        """
        lst_ret = []
        for entity in lst_entities:
            lemma, pos = entity["lemma"], entity["pos"]
            if self._lemma_lowercase:
                lemma = lemma.lower()
            key = (lemma, pos)
            self._lemma_pos_freq_so_far[key] += 1

            if self._lemma_pos_freq[key] < self._min_freq:
                continue
            if self._lemma_pos_freq_so_far[key] > self._max_freq:
                continue

            lst_ret.append(entity)

        return lst_ret

    def reset(self):
        self._lemma_pos_freq_so_far = defaultdict(int)