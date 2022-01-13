#!/usr/bin/env python
# -*- coding:utf-8 -*-
from typing import Union, List, Optional, Callable, Iterable, Dict, Any
import copy, pickle
import warnings

import os, sys, io
import bs4.element
from bs4 import BeautifulSoup
from nltk import word_tokenize
from nltk.corpus import wordnet as wn
import Levenshtein

from torch.utils.data import Dataset
from dataset_preprocessor import utils_wordnet, utils_wordnet_gloss


class WordNetGlossDataset(Dataset):

    def __init__(self, target_pos: List[str] = ["n","v"],
                 concat_extended_examples: bool = True,
                 lst_path_extended_examples_corpus: Optional[List[str]] = None,
                 filter_function: Optional[Union[Callable, List[Callable]]] = None,
                 description: str = "",
                 verbose: bool = False):
        """
        Extended gloss dataset used in SREF [Wang and Wang, EMNLP2020]
        source: https://github.com/lwmlyy/SREF
        ref: WANG, Ming; WANG, Yinglin. A Synset Relation-enhanced Framework with a Try-again Mechanism for Word Sense Disambiguation. In: Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP). 2020. p. 6229-6240.

        @param target_pos: target part-of-speeches to generate gloss sentence.
        @param concat_extended_examples: whether if we concat extended examples collected by [Wang and Wang, EMNLP2020] or not. DEFAULT: True
        @param lst_path_extended_examples_corpus: list of the path to extended example corpura pickle files.
        @param filter_function: filter function(s) to be applied for annotated gloss sentence object.
        @param verbose: output verbosity.
        """

        super().__init__()

        if concat_extended_examples:
            assert lst_path_extended_examples_corpus is not None, f"you must specify `lst_path_extended_examples_corpus`."

            if isinstance(lst_path_extended_examples_corpus, str):
                lst_path_extended_examples_corpus = [lst_path_extended_examples_corpus]

            for path in lst_path_extended_examples_corpus:
                assert os.path.exists(path), f"invalid path specified: {path}"

        self._target_pos = target_pos
        self._lst_path_extended_examples_corpus = lst_path_extended_examples_corpus
        self._concat_extended_examples = concat_extended_examples

        self._description = description

        if filter_function is None:
            self._filter_function = []
        elif isinstance(filter_function, list):
            self._filter_function = filter_function
        elif not isinstance(filter_function, list):
            self._filter_function = [filter_function]

        self._verbose = verbose

        # preload sentence object
        self._dataset = self._preload_dataset()

    def _preload_dataset(self):
        print(f"loading dataset...")
        lst_sentences = []
        for obj_sentence in self._annotated_sentence_loader():
            lst_sentences.append(obj_sentence)
        print(f"loaded annotated sentences: {len(lst_sentences)}")

        return lst_sentences

    def __len__(self):
        return len(self._dataset)

    def __iter__(self):
        for idx in range(len(self)):
            record = self.__getitem__(idx)
            yield record

    def __getitem__(self, item):
        record = self._dataset[item]
        flag = False
        for filter_function in self._filter_function:
            if filter_function(record):
                flag = True
                break
        if flag:
            return None
        return record

    def _synset_loader(self) -> wn.synset:
        for pos in self._target_pos:
            for synset in wn.all_synsets(pos=pos):
                yield synset

    def _extended_examples_loader(self) -> Dict[str, List[str]]:
        dict_synset_examples = {}
        for path in self._lst_path_extended_examples_corpus:
            print(f"loading extended examples corpus: {path}")
            ifs = io.open(path, mode="rb")
            dict_s = pickle.load(ifs)
            dict_synset_examples.update(dict_s)
            ifs.close()

        return dict_synset_examples

    def _annotated_sentence_loader(self):

        if self._concat_extended_examples:
            # dict_extended_examples: {"synset_id": ["example1", "example2", ...]}
            dict_extended_examples = self._extended_examples_loader()

        for synset in self._synset_loader():
            lst_lemmas = synset.lemmas()
            pos = synset.pos()
            synset_id = synset.name()

            # lst_lemma_surfaces = list of lemma surface forms of the synset
            lst_lemma_surfaces = list(map(utils_wordnet.lemma_to_surface_form, lst_lemmas))

            # gloss: definition sentence
            tokenized_gloss = ' '.join(word_tokenize(synset.definition()))

            # concat examples from extended examples corpora
            tokenized_examples = ""
            if self._concat_extended_examples:
                if synset_id in dict_extended_examples:
                    raw_examples = ' '.join(dict_extended_examples[synset_id])
                    tokenized_examples += ' '.join(word_tokenize(raw_examples))
            else:
                pass

            # concat synset examples from WordNet.
            # sentences are re-tokenized using word_tokenize() function.
            raw_examples = ' '.join(synset.examples())
            separator = ' ' if len(tokenized_examples) > 0 else ''
            tokenized_examples += separator + ' '.join( word_tokenize(raw_examples) )
            # for each lemma; append all lemmas, definition sentence and example sentences (may include augmented corpora)
            for lemma in lst_lemmas:
                lemma_surface = utils_wordnet.lemma_to_surface_form(lemma)
                tokenized_gloss_sentence = lemma_surface + ' - ' + ' , '.join(lst_lemma_surfaces) + ' - ' + tokenized_gloss
                if len(tokenized_examples) > 0:
                    tokenized_gloss_sentence += ' ' + tokenized_examples
                obj_annotated_sentence = self.render_tokenized_gloss_sentence_into_annotated_sentences(
                                                    lemma=lemma,
                                                    lemma_surface_form=lemma_surface,
                                                    tokenized_sentence=tokenized_gloss_sentence)

                yield obj_annotated_sentence

    @staticmethod
    def validate_annotated_sentence(obj_annotated_sentence: Dict[str, Any], verbose: bool = False):
        assert len(obj_annotated_sentence["entities"]) > 0, f"found non-annotated sentence: {obj_annotated_sentence}"

        # validate word sequence
        expected = obj_annotated_sentence["tokenized_sentence"]
        actual = " ".join(obj_annotated_sentence["words"])
        assert expected == actual, \
            f"wrong word sequence?\nexpected: {expected}\nactual: {actual}"

        lst_words = obj_annotated_sentence["words"]
        lst_entities = obj_annotated_sentence["entities"]
        for obj_entity in lst_entities:
            # validate surface form
            entity_span = lst_words[slice(*obj_entity["span"])]
            expected = obj_entity["lemma"].lower()
            actual = "_".join(entity_span).replace("-","_").lower()
            if expected != actual:
                if verbose:
                    warnings.warn(f"wrong entity span? {expected} != {actual}")

            # validate lemma key
            expected = wn.lemma_from_key(obj_entity["ground_truth_lemma_keys"][0]).name().lower()
            actual = obj_entity["lemma"].lower()
            assert expected == actual, f"wrong lemma key: {expected} != {actual}"

    def render_tokenized_gloss_sentence_into_annotated_sentences(self, lemma: wn.lemma,
                                                                 lemma_surface_form: str,
                                                                 tokenized_sentence: str):

        lst_tokens = tokenized_sentence.split(" ")
        lst_lemma_surface_tokens = lemma_surface_form.split(" ")
        entity = {
            "lemma": lemma_surface_form,
            "ground_truth_lemma_keys": [lemma.key()],
            "ground_truth_synset_ids": [lemma.synset().name()],
            "pos": lemma.synset().pos(),
            "span": [0, len(lst_lemma_surface_tokens)]
        }
        lst_surfaces = []
        for token in tokenized_sentence.split(" "):
            obj_surface = {
                "surface": token,
                "lemma": token,
                "pos": None,
                "pos_orig": None
            }
            lst_surfaces.append(obj_surface)

        # sentence object
        dict_sentence = {
            "type": "gloss",
            "tokenized_sentence": tokenized_sentence,
            "words": lst_tokens,
            "entities": [entity],
            "surfaces": lst_surfaces
        }
        return dict_sentence
