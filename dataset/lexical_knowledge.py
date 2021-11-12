#!/usr/bin/env python
# -*- coding:utf-8 -*-
import warnings
from typing import Optional, Iterable, Tuple, Set, Type, List, Dict, Callable, Union, Any
from collections import defaultdict

from torch.utils.data import Dataset
import nltk
from nltk.corpus import wordnet as wn
from .corpora import NDJSONDataset
from .filter import DictionaryFilter
from .utils import lemma_pos_to_tuple


class LemmaDataset(NDJSONDataset, Dataset):

    def __init__(self, path: str,
                 binary: bool = False,
                 lemma_lowercase: bool = True,
                 monosemous_entity_only: bool = False,
                 transform_functions=None,
                 filter_function: Optional[Union[Callable, List[Callable]]] = None,
                 n_rows: Optional[int] = None,
                 description: str = "",
                 **kwargs_for_json_loads):

        self._setup_wordnet()

        if monosemous_entity_only:
            monosemous_entity_filter = DictionaryFilter(includes={"is_monosemous":{True,}})
            if isinstance(filter_function, list):
                filter_function.append(monosemous_entity_filter)
            elif filter_function is None:
                filter_function = [monosemous_entity_filter]
            else:
                filter_function = [filter_function, monosemous_entity_filter]

        super().__init__(path, binary, transform_functions, filter_function, n_rows, description, **kwargs_for_json_loads)
        self._lemma_lowercase = lemma_lowercase
        self._monosemous_entity_only = monosemous_entity_only

        self._lexical_knowledge = self._setup_lexical_knowledge()
        self._lexical_knowledge_on_lemma_key = self._setup_lexical_knowledge_on_lemma_key()
        self._lexical_knowledge_on_synset_id = self._setup_lexical_knowledge_on_synset_id()

    def _setup_wordnet(self):
        try:
            nltk.find("corpora/wordnet")
        except:
            nltk.download("wordnet")


    def _setup_lexical_knowledge(self) -> Dict[str, Any]:
        result = {}
        for record in self:
            key = lemma_pos_to_tuple(record["lemma"], record["pos"], self._lemma_lowercase)
            result[key] = record
        return result

    def _setup_lexical_knowledge_on_lemma_key(self) -> Dict[str, Dict[str, str]]:
        result = defaultdict(dict)
        for record in self:
            iter_synset_id_lexname_pair = zip(record["synset_ids"], record["lexnames"])
            dict_synset_id_to_lexname = dict(iter_synset_id_lexname_pair)
            for lemma_key, synset_id in record["lemma_keys"].items():
                result[lemma_key]["synset_id"] = synset_id
                result[lemma_key]["lexname"] = dict_synset_id_to_lexname[synset_id]
        return result

    def _setup_lexical_knowledge_on_synset_id(self) -> Dict[str, Dict[str, List[str]]]:
        result = defaultdict(lambda : defaultdict(list))
        for record in self:
            lemma = record["lemma"]
            for lemma_key, synset_id in record["lemma_keys"].items():
                result[synset_id]["lemmas"].append(lemma)
                result[synset_id]["lemma_keys"].append(lemma_key)
        return result

    def __getitem__(self, lemma_pos: Tuple[str, str]):
        key = lemma_pos_to_tuple(lemma_pos[0], lemma_pos[1], self._lemma_lowercase)
        return self._lexical_knowledge[key]

    def __contains__(self, lemma_pos: Tuple[str, str]):
        key = lemma_pos_to_tuple(lemma_pos[0], lemma_pos[1], self._lemma_lowercase)
        return key in self._lexical_knowledge

    def get_synset_id_from_lemma_key(self, lemma_key: str):
        synset_id = self._lexical_knowledge_on_lemma_key[lemma_key]["synset_id"]
        return synset_id

    def get_lexname_from_lemma_key(self, lemma_key: str):
        lexname = self._lexical_knowledge_on_lemma_key[lemma_key]["lexname"]
        return lexname

    def get_synset_ids_from_lemma_keys(self, lemma_keys: Union[str, Iterable[str]]) -> List[str]:
        if isinstance(lemma_keys, str):
            lemma_keys = [lemma_keys]
        lst_ret = list(map(self.get_synset_id_from_lemma_key, lemma_keys))
        return lst_ret

    def get_lexnames_from_lemma_keys(self, lemma_keys: Union[str, Iterable[str]]) -> List[str]:
        if isinstance(lemma_keys, str):
            lemma_keys = [lemma_keys]
        lst_ret = list(map(self.get_lexname_from_lemma_key, lemma_keys))
        return lst_ret

    def lookup_lemma_keys_from_synset_id(self, synset_id: str):
        return self._lexical_knowledge_on_synset_id[synset_id]["lemma_keys"]

    def get_synset_codes(self, lemma: str, pos: str):
        record = self[(lemma, pos)]
        return record["synset_codes"]

    def get_synset_ids(self, lemma: str, pos: str):
        record = self[(lemma, pos)]
        return record["synset_ids"]

    def is_monosemous(self, lemma: str, pos: str):
        record = self[(lemma, pos)]
        return record["is_monosemous"]

    @property
    def synset_code_n_digits(self):
        if not hasattr(self, "_synset_code_n_digits"):
            self._synset_code_n_digits = self._apply(apply_field_name="synset_codes",
                           apply_function=lambda it_lst_codes: max([max(map(len, lst_codes)) for lst_codes in it_lst_codes]))
        return self._synset_code_n_digits

    @property
    def synset_code_n_ary(self):
        if not hasattr(self, "_synset_code_n_ary"):
            self._synset_code_n_ary = self._apply(apply_field_name="synset_codes",
                           apply_function=lambda it_lst_codes: max([max(map(max, lst_codes)) for lst_codes in it_lst_codes]) + 1)
        return self._synset_code_n_ary

    @property
    def n_entity(self):
        return len(self._lexical_knowledge)

    @property
    def lemmas(self):
        return self.distinct_values(column="lemma")

    @property
    def lemma_and_pos(self):
        return set(self._lexical_knowledge.keys())

    @property
    def entities(self):
        return self.lemma_and_pos

    @property
    def pos_tagset(self):
        return self.distinct_values(column="pos")

    @property
    def n_lemma(self):
        return len(self.lemmas)

    @property
    def n_pos(self):
        return len(self.pos_tagset)

    @property
    def n_lemma_and_pos(self):
        return len(self.lemma_and_pos)

    @property
    def verbose(self):
        v = super().verbose
        v["synset_code_n_digits"] = self.synset_code_n_digits
        v["synset_code_n_ary"] = self.synset_code_n_ary
        v["n_lemma_and_pos"] = self.n_lemma_and_pos
        v["n_lemma"] = self.n_lemma
        v["n_pos"] = self.n_pos
        v["monosemous_entity_only"] = self._monosemous_entity_only
        return v


class SynsetDataset(NDJSONDataset, Dataset):

    def __init__(self, path: str,
                 binary: bool = False,
                 lemma_lowercase: bool = True,
                 transform_functions=None,
                 filter_function: Optional[Union[Callable, List[Callable]]] = None,
                 n_rows: Optional[int] = None,
                 description: str = "",
                 **kwargs_for_json_loads):

        self._setup_wordnet()

        if lemma_lowercase:
            if transform_functions is None:
                transform_functions = {}
            transform_functions["lemmas"] = self._lowercase

        super().__init__(path, binary, transform_functions, filter_function, n_rows, description, **kwargs_for_json_loads)
        self._lemma_lowercase = lemma_lowercase

        self._synsets = self._setup_lexical_knowledge()


    def _setup_wordnet(self):
        try:
            nltk.find("corpora/wordnet")
        except:
            nltk.download("wordnet")

    def _lowercase(self, lst_strings: List[str]):
        return [string.lower() for string in lst_strings]

    def _synset_code_to_lookup_key(self, synset_code: List[int]):
        return "-".join(map(str, synset_code))

    def _setup_lexical_knowledge(self) -> Dict[str, Any]:
        result = {}
        for record in self:

            key_id = record["id"]
            key_code = self._synset_code_to_lookup_key(record["code"])
            result[key_id] = record
            result[key_code] = record

        return result

    def __getitem__(self, synset_id_or_synset_code: Union[str, List[int]]):
        if isinstance(synset_id_or_synset_code, str):
            return self._synsets[synset_id_or_synset_code]
        elif isinstance(synset_id_or_synset_code, list):
            key = self._synset_code_to_lookup_key(synset_id_or_synset_code)
            return self._synsets[key]
        else:
            raise ValueError(f"unknown key type: {type(synset_id_or_synset_code)}")

    def __contains__(self, synset_id_or_synset_code: Union[str, List[int]]):
        try:
            self.__getitem__(synset_id_or_synset_code)
            return True
        except:
            return False

    def get_parent_synset(self, synset_id: str):
        parent_synset_id = self[synset_id].get("parent_synset_id", None)
        if parent_synset_id is None:
            return None
        else:
            if parent_synset_id in self:
                return self[parent_synset_id]
            else:
                return None

    def get_ancestor_synsets(self, synset_id: str):
        lst_ancestors = []
        while True:
            parent_synset = self.get_parent_synset(synset_id)
            if parent_synset is None:
                return lst_ancestors
            else:
                lst_ancestors.append(parent_synset)
                synset_id = parent_synset["id"]

    def get_synset_code(self, synset_id: str):
        return self[synset_id]["code"]