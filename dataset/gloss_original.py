#!/usr/bin/env python
# -*- coding:utf-8 -*-
from typing import Union, List, Optional, Callable, Iterable, Dict, Any
import copy, pickle
import warnings

import os, sys, io
import bs4.element
from bs4 import BeautifulSoup
from nltk.corpus import wordnet as wn
import Levenshtein

from torch.utils.data import Dataset
from dataset_preprocessor import utils_wordnet, utils_wordnet_gloss

class WordNetGlossDataset(Dataset):

    def __init__(self, lst_path_gloss_corpus: Union[str, List[str]],
                 concat_hypernym_definition_sentence: bool = False,
                 filter_function: Optional[Union[Callable, List[Callable]]] = None,
                 description: str = "",
                 verbose: bool = False):

        super().__init__()

        if isinstance(lst_path_gloss_corpus, str):
            lst_path_gloss_corpus = [lst_path_gloss_corpus]
        for path in lst_path_gloss_corpus:
            assert os.path.exists(path), f"invalid path specified: {path}"

        self._lst_path_gloss_corpus = lst_path_gloss_corpus
        self._concat_hypernym_definition_sentence = concat_hypernym_definition_sentence

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

    def _synset_node_loader(self) -> Iterable[bs4.element.Tag]:
        for path in self._lst_path_gloss_corpus:
            ifs = io.open(path, mode="r")
            soup = BeautifulSoup(ifs, features="lxml")

            css_selector = "wordnet synset"
            for synset_node in soup.select(css_selector):
                yield synset_node

            ifs.close()

    def _annotated_sentence_loader(self):

        dict_synset_definition = {}
        if self._concat_hypernym_definition_sentence:
            # preload definition sentence of all synsets.
            for synset_node in self._synset_node_loader():
                pos = synset_node.get("pos")
                synset_offset = int(synset_node.get("ofs"))
                synset = wn.synset_from_pos_and_offset(pos, synset_offset)
                synset_id = synset.name()

                # extract definition sentence
                assert len(synset_node.select("def")) == 1, f"missing definition node: {synset_node}"
                lst_def_surfaces = self._parse_definition_node_to_surfaces(definition_node=synset_node.select_one("def"))
                dict_synset_definition[synset_id] = lst_def_surfaces

        for synset_node in self._synset_node_loader():
            pos = synset_node.get("pos")
            synset_offset = int(synset_node.get("ofs"))
            synset = wn.synset_from_pos_and_offset(pos, synset_offset)
            synset_id = synset.name()

            # get hypernym definition sentence
            if self._concat_hypernym_definition_sentence:
                lst_hypernyms = wn.synset(synset_id).hypernyms()
                if len(lst_hypernyms) > 0:
                    hypernym_synset_id = lst_hypernyms[0].name()
                    lst_hypernym_def_surfaces = dict_synset_definition[hypernym_synset_id]
                else:
                    lst_hypernym_def_surfaces = None
            else:
                lst_hypernym_def_surfaces = None

            # lemma information
            lst_lemmas = []
            lst_terms = [term.text for term in synset_node.select("terms term")]
            assert len(lst_terms) == len(synset.lemmas()), f"lemma lookup failed: {synset_id}, {lst_terms} != {synset.lemmas()}"
            for term, lemma in zip(lst_terms, synset.lemmas()):
                dict_lemma = {
                    "lemma_key": lemma.key(),
                    "lemma": lemma.name(),
                    "surface": term,
                    "words": term.split(" "),
                }
                # lemmaとtermの違いは，wordnet.lemmas(...)用にformatされているか否か
                assert lemma.name().split("%")[0] == term.replace(" ","_")
                try:
                    _ = wn.lemma_from_key(dict_lemma["lemma_key"])
                except Exception as e:
                    warnings.warn(f"invalid lemma key: {e}")
                    continue
                lst_lemmas.append(dict_lemma)

            # extract definition sentence
            assert len(synset_node.select("def")) == 1, f"missing definition node: {synset_node}"
            lst_def_surfaces = self._parse_definition_node_to_surfaces(definition_node=synset_node.select_one("def"))

            # then concat lemmas into definition in order to create sense-annotated sentence.
            lst_def_sentences = self.render_lemmas_and_definition_into_annotated_sentences(pos=pos,
                                                                                           synset_id=synset_id, lst_lemmas=lst_lemmas,
                                                                                           lst_def_surfaces=lst_def_surfaces,
                                                                                           lst_hypernym_def_surfaces=lst_hypernym_def_surfaces)

            # parse example nodes
            lst_example_sentences = []
            for example_node in synset_node.select("ex"):
                if len(example_node.select("wf id")) == 0:
                    if self._verbose:
                        print(f"skip non-annotated example: {synset_id}|{example_node.get('id')}")
                    continue
                lst_obj_sentence = self._parse_example_node_into_annotated_sentence(pos=pos, synset_id=synset_id,
                                                                                lst_lemmas=lst_lemmas,
                                                                                example_node=example_node,
                                                                                lst_hypernym_def_surfaces=lst_hypernym_def_surfaces)

                for obj_sentence in lst_obj_sentence:
                    if len(obj_sentence["entities"]) == 0:
                        if self._verbose:
                            print(f"skip non-annotated example: {synset_id}|{example_node.get('id')}")
                        continue
                    lst_example_sentences.append(obj_sentence)

            lst_annotated_sentences = lst_def_sentences + lst_example_sentences

            for obj_annotated_sentence in lst_annotated_sentences:
                self.validate_annotated_sentence(obj_annotated_sentence, verbose=self._verbose)
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

    # extract surfaces from definition subtree
    def _parse_definition_node_to_surfaces(self, definition_node: bs4.element.Tag) -> List[Dict[str, str]]:
        lst_surface = []
        for surface_node in definition_node.select("wf"):
            if not surface_node.has_attr("lemma"):
                assert surface_node.get("tag") == "ignore", f"unexpected node: {surface_node}"
                continue
            pos_orig = surface_node.get("pos")
            pos = utils_wordnet.ptb_tagset_to_wordnet_tagset(pos_orig, na_value="o")
            dict_surface = {
                "surface": utils_wordnet_gloss.clean_up_surface(surface_node.text),
                "lemma": utils_wordnet_gloss.clean_up_lemma(surface_node.get("lemma")),
                "pos_orig": pos_orig,
                "pos": pos
            }
            lst_surface.append(dict_surface)

        return lst_surface

    def render_lemmas_and_definition_into_annotated_sentences(self, pos: str, synset_id: str,
                                                              lst_lemmas: List[Dict[str, Union[str, List[str]]]],
                                                              lst_def_surfaces: List[Dict[str, str]],
                                                              lst_hypernym_def_surfaces: Optional[List[Dict[str, str]]] = None):
        lst_sentences = []
        lst_def_words = [surface["surface"] for surface in lst_def_surfaces]
        for dict_lemma in lst_lemmas:
            entity = {
                "lemma": dict_lemma["lemma"],
                "ground_truth_lemma_keys": [dict_lemma["lemma_key"]],
                "ground_truth_synset_ids": [synset_id],
                "pos": pos,
                "span": [0, len(dict_lemma["words"])]
            }
            obj_surface = {
                "surface": dict_lemma["surface"],
                "lemma": dict_lemma["lemma"],
                "pos": pos,
                "pos_orig": None
            }

            # synset lemma と 語釈文を単純連結
            dict_sentence = {
                "type": "definition",
                "tokenized_sentence": " ".join([dict_lemma["surface"]] + lst_def_words),
                "words": dict_lemma["words"] + lst_def_words,
                "entities": [entity],
                "surfaces": [obj_surface] + lst_def_surfaces
            }
            lst_sentences.append(dict_sentence)

            # (optional) hypernym definition sentenceを単純連結
            if lst_hypernym_def_surfaces is not None:
                dict_sentence_ = copy.deepcopy(dict_sentence)
                dict_sentence_["type"] = "definition+hypernym.def"
                dict_sentence_["words"] += [surface["surface"] for surface in lst_hypernym_def_surfaces]
                dict_sentence_["tokenized_sentence"] = " ".join(dict_sentence_["words"])
                dict_sentence_["surfaces"] += lst_hypernym_def_surfaces
                lst_sentences.append(dict_sentence_)

        return lst_sentences

    # extract and annotate surfaces from example subtree
    def _parse_example_node_into_annotated_sentence(self, pos: str, synset_id: str, lst_lemmas: List[Dict[str, Union[str, List[str]]]],
                                                    example_node: bs4.element.Tag,
                                                    lst_hypernym_def_surfaces: Optional[List[Dict[str, str]]] = None) -> List[Dict[str, Any]]:
        lst_sentences = []
        lst_words = []
        lst_surfaces = []; lst_entities = []
        possible_lemma_keys = {lemma["lemma_key"] for lemma in lst_lemmas}
        for surface_node in example_node.select("wf,glob"):
            if not surface_node.has_attr("lemma"):
                assert surface_node.get("tag") == "ignore", f"unexpected node: {surface_node}"
                continue

            if surface_node.name == "wf":
                surface = utils_wordnet_gloss.clean_up_surface(surface_node.text)
            elif surface_node.name == "glob":
                surface = surface_node.select_one("id").get("lemma")
            else:
                raise ValueError(f"unexpected surface node: {surface_node}")

            # warning: surface may be Multi-Word Expression.
            dict_surface = {
                "surface": surface,
                "lemma": utils_wordnet_gloss.clean_up_lemma(surface_node.get("lemma")),
                "pos_orig": None,
                "pos": None
            }

            surface_words = dict_surface["surface"].split(" ")

            # check if current surface is sense-annotated entity.
            id_node = surface_node.select_one("id")
            if id_node is not None:
                lemma_key = id_node.get("sk")
                if lemma_key in possible_lemma_keys:
                    lemma = wn.lemma_from_key(lemma_key).name()
                    entity = {
                        "lemma": lemma,
                        "ground_truth_lemma_keys": [lemma_key],
                        "ground_truth_synset_ids": [synset_id],
                        "pos": pos,
                        "span": [len(lst_words), len(lst_words) + len(surface_words)]
                    }
                    dict_surface["lemma"] = lemma # overwrite lemma from the lemma_key lookup result.
                    dict_surface["pos"] = pos
                    lst_entities.append(entity)
                else:
                    if self._verbose:
                        warnings.warn(f"invalid lemma key: {lemma_key}")

            lst_surfaces.append(dict_surface)

            lst_words = lst_words + surface_words

        assert len(lst_surfaces) > 0, f"empty sentence: {example_node}"
        if len(lst_entities) == 0:
            if self._verbose:
                warnings.warn(f"no entity has found: {example_node}")

        dict_sentence = {
            "type": "example",
            "tokenized_sentence": " ".join([dict_surface["surface"] for dict_surface in lst_surfaces]),
            "words": lst_words,
            "entities": lst_entities,
            "surfaces": lst_surfaces
        }
        lst_sentences.append(dict_sentence)

        if lst_hypernym_def_surfaces is not None:
            dict_sentence_ = copy.deepcopy(dict_sentence)
            dict_sentence_["type"] = "example+hypernym.def"
            dict_sentence_["words"] += [surface["surface"] for surface in lst_hypernym_def_surfaces]
            dict_sentence_["tokenized_sentence"] = " ".join(dict_sentence_["words"])
            dict_sentence_["surfaces"] += lst_hypernym_def_surfaces
            lst_sentences.append(dict_sentence_)

        return lst_sentences


class ExtendedWordNetGlossDataset(Dataset):

    def __init__(self, lst_path_gloss_corpus: Union[str, List[str]],
                 filter_function: Optional[Union[Callable, List[Callable]]] = None,
                 description: str = "",
                 verbose: bool = False):
        """
        Extended gloss dataset used in SREF [Wang and Wang, EMNLP2020]
        source: https://github.com/lwmlyy/SREF
        ref: WANG, Ming; WANG, Yinglin. A Synset Relation-enhanced Framework with a Try-again Mechanism for Word Sense Disambiguation. In: Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP). 2020. p. 6229-6240.

        @param lst_path_gloss_corpus: list of the path to pickle files.
        @param filter_function: filter function(s) to be applied for annotated sentence object.
        @param verbose: output verbosity.
        """
        super().__init__()

        if isinstance(lst_path_gloss_corpus, str):
            lst_path_gloss_corpus = [lst_path_gloss_corpus]
        for path in lst_path_gloss_corpus:
            assert os.path.exists(path), f"invalid path specified: {path}"

        self._lst_path_gloss_corpus = lst_path_gloss_corpus

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

    def _pickle_loader(self) -> Iterable[Dict[str, str]]:
        for path in self._lst_path_gloss_corpus:
            ifs = io.open(path, mode="rb")
            dict_texts = pickle.load(ifs)

            for synset_id, lst_sentences in dict_texts.items():
                for sentence in lst_sentences:
                    ret = {
                        "synset_id": synset_id,
                        "sentence": sentence
                    }
                    yield ret

            ifs.close()

    def _annotate_sentence(self, synset_id: str, sentence: str) -> List[Dict[str, Any]]:
        lst_obj_sentences = []

        synset = wn.synset(synset_id)
        pos = synset.pos()
        lemmas = synset.lemmas()
        dict_lemmas = {lemma.name():lemma.key() for lemma in lemmas}
        # assert len(dict_lemmas) == len(lemmas), f"duplicate lemma? {lemmas}"

        # create annotated sentence object
        # we just omit `surfaces` object because both PoS and baseform aren't available.
        dict_sentence = {
            "type": "extended.example",
            "tokenized_sentence": sentence,
            "words": sentence.split(" "),
            "entities": [],
            "surfaces": None
        }

        ### sense annotation ###

        # 1. test if beginnig-of-sentence is the lemma of synset.
        n_entity_span = synset_id.count("_") + 1
        target_ = sentence.split(" ")[0:n_entity_span]
        target = " ".join(target_)
        matched_lemma, edit_distance = self.select_most_similar_string(target=target, lst_candidates=list(dict_lemmas.keys()))

        src_char_length = min(len(target), len(matched_lemma))
        if edit_distance / src_char_length > 0.5:
            prepend_lemma = True
        else:
            prepend_lemma = False

        # 2. if prepend_lemma = True, then prepend lemma as the sense-annotated entity.
        if prepend_lemma:
            for lemma, lemma_key in dict_lemmas.items():
                surface_words = lemma.split("_")
                entity = {
                    "lemma": lemma,
                    "ground_truth_lemma_keys": [lemma_key],
                    "ground_truth_synset_ids": [synset_id],
                    "pos": pos,
                    "span": [0, len(surface_words)]
                }
                dict_sentence_ = copy.deepcopy(dict_sentence)
                dict_sentence_["tokenized_sentence"] = lemma.replace("_"," ") + " " + sentence
                dict_sentence_["words"] = surface_words + dict_sentence_["words"]
                dict_sentence_["entities"] = [entity]
                lst_obj_sentences.append(dict_sentence_)

        # 3. if prepend_lemma = False, then markup beginnig-of-sentence as the sense-annotated entity.
        else:
            lemma_key = dict_lemmas[matched_lemma]
            entity = {
                "lemma": matched_lemma,
                "ground_truth_lemma_keys": [lemma_key],
                "ground_truth_synset_ids": [synset_id],
                "pos": pos,
                "span": [0, n_entity_span]
            }
            dict_sentence_ = copy.deepcopy(dict_sentence)
            dict_sentence_["entities"] = [entity]
            lst_obj_sentences.append(dict_sentence_)

        return lst_obj_sentences

    def _annotated_sentence_loader(self) -> Dict[str, Any]:
        for record in self._pickle_loader():
            lst_obj_sentence = self._annotate_sentence(synset_id=record["synset_id"], sentence=record["sentence"])
            for obj_sentence in lst_obj_sentence:
                # WordNetGlossDataset.validate_annotated_sentence(obj_sentence)
                yield obj_sentence

    @staticmethod
    def select_most_similar_string(target: str, lst_candidates: List[str]):
        lst_edit_distances = [Levenshtein.distance(target, candidate) for candidate in lst_candidates]
        # sort by edit distance (ascending order)
        lst_tup_sorted = sorted(zip(lst_edit_distances, lst_candidates))
        # take most similar one
        edit_distance, candidate = lst_tup_sorted[0]
        return candidate, edit_distance

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
