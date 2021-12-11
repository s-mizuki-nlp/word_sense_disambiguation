#!/usr/bin/env python
# -*- coding:utf-8 -*-

from typing import Union, List, Optional, Callable, Iterable, Dict, Any
import warnings

import os, sys, io
import bs4.element
from bs4 import BeautifulSoup
from nltk.corpus import wordnet as wn

from ._base import AbstractFormatDataset
from torch.utils.data import Dataset
from dataset_preprocessor import utils_wordnet, utils_wordnet_gloss

class WordNetGlossDataset(Dataset):

    def __init__(self, lst_path_gloss_corpus: Union[str, List[str]],
                 transform_functions = None,
                 filter_function: Optional[Union[Callable, List[Callable]]] = None,
                 entity_filter_function: Optional[Union[Callable, List[Callable]]] = None,
                 description: str = "",
                 verbose: bool = False):

        super().__init__(path=lst_path_gloss_corpus, transform_functions=transform_functions,
                         filter_function=filter_function, n_rows=None, description=description)

        if isinstance(lst_path_gloss_corpus, str):
            lst_path_gloss_corpus = [lst_path_gloss_corpus]
        for path in lst_path_gloss_corpus:
            assert os.path.exists(path), f"invalid path specified: {path}"

        self._lst_path_gloss_corpus = lst_path_gloss_corpus

        self._description = description
        self._transform_functions = transform_functions

        if filter_function is None:
            self._filter_function = []
        elif isinstance(filter_function, list):
            self._filter_function = filter_function
        elif not isinstance(filter_function, list):
            self._filter_function = [filter_function]

        if entity_filter_function is None:
            self._entity_filter_function = []
        elif isinstance(entity_filter_function, list):
            self._entity_filter_function = entity_filter_function
        elif not isinstance(entity_filter_function, list):
            self._entity_filter_function = [entity_filter_function]

        self._verbose = verbose

    def __iter__(self):
        if isinstance(self._transform_functions, dict):
            for field_name, function in self._transform_functions.items():
                if hasattr(function, "reset"):
                    function.reset()

        iter_records = self._annotated_sentence_loader()
        n_read = 0
        for record in iter_records:
            yield record
            n_read += 1

    def _synset_node_loader(self) -> Iterable[bs4.element.Tag]:
        for path in self._lst_path_gloss_corpus:
            ifs = io.open(path, mode="r")
            soup = BeautifulSoup(ifs, features="lxml")

            css_selector = "wordnet synset"
            for synset_node in soup.select(css_selector):
                yield synset_node

            ifs.close()

    def _annotated_sentence_loader(self):

        for synset_node in self._synset_node_loader():
            pos = synset_node.get("pos")
            synset_offset = int(synset_node.get("ofs"))
            synset = wn.synset_from_pos_and_offset(pos, synset_offset)
            synset_id = synset.name()

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
                assert lemma.name().split("%")[0] == term.replace(" ","_")
                lst_lemmas.append(dict_lemma)

            # extract definition sentence
            assert len(synset_node.select("def")) == 1, f"missing definition node: {synset_node}"
            lst_def_surfaces = self._parse_definition_node_to_surfaces(definition_node=synset_node.select_one("def"))

            # then concat lemmas into definition in order to create sense-annotated sentence.
            lst_def_sentences = self.render_lemmas_and_definition_into_annotated_sentences(pos=pos,
                                                                                           synset_id=synset_id, lst_lemmas=lst_lemmas,
                                                                                           lst_def_surfaces=lst_def_surfaces)

            # parse example nodes
            lst_example_sentences = []
            for example_node in synset_node.select("ex"):
                if len(example_node.select("id")) == 0:
                    print(f"skip non-annotated example: {synset_id}|{example_node.get('id')}")
                    continue
                obj_sentence = self._parse_example_node_into_annotated_sentence(pos=pos, synset_id=synset_id,
                                                                                lst_lemmas=lst_lemmas,
                                                                                example_node=example_node)
                if len(obj_sentence["entities"]) == 0:
                    print(f"skip non-annotated example: {synset_id}|{example_node.get('id')}")
                lst_example_sentences.append(obj_sentence)

            lst_annotated_sentences = lst_def_sentences + lst_example_sentences

            for obj_annotated_sentence in lst_annotated_sentences:
                yield obj_annotated_sentence

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
                                                              lst_def_surfaces: List[Dict[str, str]]):
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
        return lst_sentences

    # extract and annotate surfaces from example subtree
    def _parse_example_node_into_annotated_sentence(self, pos: str, synset_id: str, lst_lemmas: List[Dict[str, Union[str, List[str]]]],
                                                    example_node: bs4.element.Tag) -> Dict[str, Any]:
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
                surface = utils_wordnet_gloss.lemma_to_surface(surface_node.get("lemma"))
            else:
                raise ValueError(f"unexpected surface node: {surface_node}")

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
                    entity = {
                        "lemma": dict_surface["lemma"],
                        "ground_truth_lemma_keys": [lemma_key],
                        "ground_truth_synset_ids": [synset_id],
                        "pos": pos,
                        "span": [len(lst_words), len(lst_words) + len(surface_words)]
                    }
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

        return dict_sentence