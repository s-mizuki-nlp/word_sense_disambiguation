#!/usr/bin/env python
# -*- coding:utf-8 -*-
import copy
import io, os, json
from typing import Union, Collection, Optional, Dict, Any, Iterable, Callable, List
from torch.utils.data import Dataset

import bs4.element
from bs4 import BeautifulSoup
from nltk.corpus import wordnet as wn

from dataset_preprocessor import utils_wordnet


class WSDEvaluationDataset(Dataset):

    def __init__(self, path_corpus: str,
                 path_ground_truth_labels: str,
                 lookup_candidate_senses: bool = True,
                 num_concat_surrounding_sentences: Optional[int] = None,
                 transform_functions = None,
                 filter_function: Optional[Union[Callable, List[Callable]]] = None,
                 entity_filter_function: Optional[Union[Callable, List[Callable]]] = None,
                 description: str = ""):

        """
        [Raganato+, 2017]が提供するWord Sense Disambiguation Evaluation Datasetを読み込むクラス
        WordNetを参照して，正解候補(lemma id, lemma x pos, synset)も生成する．

        [Raganato+, 2017] Word Sense Disambiguation: A Unified Evaluation Framework and Empirical Comparison
        http://lcl.uniroma1.it/wsdeval/

        :param path_corpos: コーパスのパス (`*.data.xml`)
        :param path_ground_truth_labels: 正解ラベルのパス(`*.gold.key.txt`)
        :param num_concat_surrounding_sentences: 同一文書内の前後n文を連結する．Noneの場合は無効．DEFAULT: None
        :param transform_functions: データ変形定義，Dictionaryを指定．keyはフィールド名，valueは変形用関数
        :param filter_function: 除外するか否かを判定する関数
        :param description: 説明
        """

        super().__init__()

        assert os.path.exists(path_corpus), f"invalid path specified: {path_corpus}"
        self._path = path_corpus
        self._path_ground_truth_labels = path_ground_truth_labels

        self._ground_truth_labels = self._load_ground_truth_labels(path=path_ground_truth_labels)
        self._lookup_candidate_senses = lookup_candidate_senses
        self._num_concat_surrounding_sentences = 0 if num_concat_surrounding_sentences is None else num_concat_surrounding_sentences

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

        self._n_sample = None

        self._records = self._preload_sentences()

    def _load_ground_truth_labels(self, path: str):
        dict_labels = {}
        with io.open(path, mode="r") as ifs:
            for entry in ifs:
                lst_ = entry.strip("\n").split(" ")
                key, labels = lst_[0], lst_[1:]
                dict_labels[key] = labels
        return dict_labels

    def _entity_filter(self, lst_entities: List[Dict[str, Any]]):
        lst_ret = []
        for entity in lst_entities:
            is_filtered_entity = any([filter_function(entity) for filter_function in self._entity_filter_function])
            if not is_filtered_entity:
                lst_ret.append(entity)
        return lst_ret

    def _lookup_candidate_senses_from_wordnet(self, lemma: str, pos: str) -> List[Dict[str, str]]:
        lst_lemmas = wn.lemmas(lemma, pos=pos)
        assert len(lst_lemmas) > 0, f"unknown lemma: {lemma}|{pos}"

        lst_candidates = []
        for lemma in lst_lemmas:
            dict_candidate = {
                "lemma_key": lemma.key(),
                "synset": lemma.synset().name(),
                "lexname": lemma.synset().lexname(),
            }
            lst_candidates.append(dict_candidate)
        return lst_candidates

    def _parse_instance_node(self, instance_node: bs4.element.Tag):
        _id = instance_node.get("id")
        dict_output = {
            "id": _id,
            "lemma": instance_node.get("lemma"),
            "pos_orig": instance_node.get("pos"),
            "pos": utils_wordnet.universal_tagset_to_wordnet_tagset(instance_node.get("pos"))
        }
        lst_lemma_keys = self._ground_truth_labels[_id]
        lst_synset_ids = [wn.lemma_from_key(lemma_key).synset().name() for lemma_key in lst_lemma_keys]
        dict_output["ground_truth_lemma_keys"] = lst_lemma_keys
        dict_output["ground_truth_synset_ids"] = lst_synset_ids

        if self._lookup_candidate_senses:
            dict_output["candidate_senses"] = self._lookup_candidate_senses_from_wordnet(lemma=dict_output["lemma"], pos=dict_output["pos"])

        return dict_output

    def _extract_sentence_ids(self, sentence_id: str, delimiter: str = "."):
        lst_ids = sentence_id.split(delimiter)
        corpus_id = lst_ids[0]
        document_id = delimiter.join(lst_ids[:2])
        sentence_id = delimiter.join(lst_ids[:3])
        return corpus_id, document_id, sentence_id

    def _parse_sentence_node(self, sentence_node: bs4.element.Tag):

        # extract words
        lst_words = []
        lst_entities = []
        lst_surfaces = []
        node_surfaces = sentence_node.select("wf,instance")
        for node_surface in node_surfaces:
            words = node_surface.text.split(" ")
            span = [len(lst_words), len(lst_words) + len(words)]
            lst_words += words

            # extract wsd entity
            if node_surface.name == "instance":
                dict_instance = self._parse_instance_node(node_surface)
                dict_instance["span"] = span
                lst_entities.append(dict_instance)

            dict_surface = {
                "surface": node_surface.text,
                "lemma": node_surface.get("lemma"),
                "pos": node_surface.get("pos")
            }
            lst_surfaces.append(dict_surface)

        # extract ids and text
        sentence_id = sentence_node.get("id")
        corpus_id, document_id, sentence_id = self._extract_sentence_ids(sentence_id)
        text = sentence_node.text.strip().replace("\n", " ")

        dict_sentence = {
            "corpus_id": corpus_id,
            "document_id": document_id,
            "sentence_id": sentence_id,
            "tokenized_sentence": text,
            "words": lst_words,
            "entities": lst_entities,
            "surfaces": lst_surfaces
        }
        return dict_sentence

    @staticmethod
    def concat_sentence_objects(source_sentence, lst_concat_sentences):
        source_sentence_ = copy.deepcopy(source_sentence)
        for concat_sentence in lst_concat_sentences:
            # append tokenized sentence string
            source_sentence_["tokenized_sentence"] += " " + concat_sentence["tokenized_sentence"]
            # extend word sequence
            source_sentence_["words"] += concat_sentence["words"]
            # extend surface sequence
            source_sentence_["surfaces"] += concat_sentence["surfaces"]

        return source_sentence_

    def _sentence_loader(self) -> Iterable[Dict[str, Any]]:
        ifs = io.open(self._path, mode="r")
        soup = BeautifulSoup(ifs, features="lxml")

        css_selector = "corpus text sentence"
        for sentence_node in soup.select(css_selector):
            yield self._parse_sentence_node(sentence_node)

        ifs.close()

    def _preload_sentences(self) -> List[Dict[str, Any]]:
        lst_sentences = []
        for record in self._sentence_loader():
            lst_sentences.append(record)
        return lst_sentences

    def get_record(self, idx: int):
        offset = self._num_concat_surrounding_sentences
        record = self._records[idx]
        # (optional) concatenate surrounding sentences (previous and next N sentences)
        if offset > 0:
            document_id = record["document_id"]
            lst_surrounding_sentences = self._records[max(0, idx-offset):idx]
            lst_surrounding_sentences += self._records[idx+1:idx+offset+1]
            lst_surrounding_sentences = [record for record in lst_surrounding_sentences if record["document_id"] == document_id]
            assert len(lst_surrounding_sentences) > 0, f"something went wrong: {record}"
            record = self.concat_sentence_objects(source_sentence=record, lst_concat_sentences=lst_surrounding_sentences)

        return record

    def _filter_transform_records(self):
        for idx in range(len(self._records)):
            record = self.get_record(idx)

            # transform each field of the entry
            entry = self._transform(record)
            # verify the entry is valid or not
            if self._filter(entry) == True:
                continue
            # filter entities
            record["entities"] = self._entity_filter(record["entities"])

            yield record

    def _apply(self, apply_field_name: str, apply_function: Callable, na_value: Optional[Any] = None):

        _transform_cache = self._transform_functions
        self._transform_functions = None

        it = (entry.get(apply_field_name, na_value) for entry in self)
        ret =  apply_function(filter(bool, it))

        self._transform_functions = _transform_cache
        return ret

    def distinct_values(self, column: str) -> List[str]:
        return self._apply(apply_field_name=column, apply_function=lambda it: list(set(it)))

    def iter_specific_field(self, field_name: str, na_value: Optional[Any] = None):
        for entry in self:
            yield entry.get(field_name, na_value)

    def __len__(self):
        if self._n_sample is not None:
            return self._n_sample

        n_sample = 0
        for _ in self:
            n_sample += 1

        self._n_sample = n_sample

        return n_sample

    def _transform(self, entry: Dict[str, Any]):
        if self._transform_functions is None:
            return entry

        for field_name, transform_function in self._transform_functions.items():
            entry[field_name] = transform_function(entry[field_name])

        return entry

    def _filter(self, entry: Dict[str, Any]):
        for filter_function in self._filter_function:
            if filter_function(entry) == True:
                return True
        return False

    def __getitem__(self, index):
        return self.get_record(index)

    def __iter__(self):
        if isinstance(self._transform_functions, dict):
            for field_name, function in self._transform_functions.items():
                if hasattr(function, "reset"):
                    function.reset()

        iter_records = self._filter_transform_records()
        n_read = 0
        for record in iter_records:
            yield record

            n_read += 1

    @property
    def verbose(self):
        ret = {
            "path": (self._path, self._path_ground_truth_labels),
            "nrows": self.__len__(),
            "description": self._description,
            "filter_function": self._filter_function,
            "transform_functions": self._transform_functions
        }
        return ret
