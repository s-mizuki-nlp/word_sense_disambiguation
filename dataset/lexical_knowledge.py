#!/usr/bin/env python
# -*- coding:utf-8 -*-
import warnings
from typing import Optional, Iterable, Tuple, Set, Type, List, Dict, Callable, Union, Any
from collections import defaultdict, Counter

import progressbar
from torch.utils.data import Dataset
import nltk
from nltk.corpus import wordnet as wn
from .corpora import NDJSONDataset
from .filter import DictionaryFilter
from .utils import lemma_pos_to_tuple, sequence_to_str


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
        result = {}
        for record in self:
            lemma = record["lemma"]
            for lemma_key, synset_id in record["lemma_keys"].items():
                if synset_id not in result:
                    result[synset_id] = defaultdict(list)
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
        v["n_lemma_and_pos"] = self.n_lemma_and_pos
        v["n_lemma"] = self.n_lemma
        v["pos_tagset"] = self.pos_tagset
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
        self._sense_code_taxonomy = self._setup_sense_code_taxonomy()
        self._sense_code_taxonomy_by_index = {record["idx"]:key for key, record in self._sense_code_taxonomy.items()}

    def _setup_wordnet(self):
        try:
            nltk.find("corpora/wordnet")
        except:
            nltk.download("wordnet")

    def _lowercase(self, lst_strings: List[str]):
        return [string.lower() for string in lst_strings]

    @staticmethod
    def sequence_to_str(sequence: List[int]):
        if len(sequence) == 0:
            raise ValueError(f"`sequence` cannot be empty.")
        else:
            return sequence_to_str(sequence)

    def _trim_trailing_zeroes(self, sense_code: List[int]) -> List[int]:
        n_length = len(sense_code) - sense_code.count(0)
        return sense_code[:n_length]

    def _transform_prefix_to_sense_code_and_pos(self, sense_code_prefix: List[int]) -> Tuple[List[int], str]:
        pos_idx = sense_code_prefix[0]
        pos = self._pos_index_rev[pos_idx]
        partial_sense_code = sense_code_prefix[1:]

        n_pad = self.n_digits - len(partial_sense_code)
        padded = partial_sense_code + [0]*n_pad
        return padded, pos

    def _setup_lexical_knowledge(self) -> Dict[str, Any]:
        result = {}
        for record in self:

            key_id = record["id"]
            key_code = self.sequence_to_str(record["code"])
            result[key_id] = record
            result[key_code] = record

        return result

    def _setup_sense_code_taxonomy(self) -> Dict[str, Any]:
        result = {}

        # 1. count ancestors
        num_descendents = Counter()
        set_pos = set()
        for record in self:
            pos = record["pos"]
            sense_code = record["code"]
            prefixes = self.synset_code_to_prefixes(sense_code, pos=pos)
            str_prefixes = [self.sequence_to_str(prefix) for prefix in prefixes]
            num_descendents.update(str_prefixes)
            set_pos.add(pos)
        # decrement oneself except of virtual root (lookup_key={"n","v"})
        root_keys = {str(self.pos_index[pos]) for pos in set_pos}
        for key, value in num_descendents.items():
            if key in root_keys:
                continue
            num_descendents[key] = value - 1

        # 2. record sense code prefix info
        index = 1
        for record in self:
            pos = record["pos"]
            sense_code = record["code"]
            # ([1,2,0,0],'n') -> [[65,], [65,1], [65,1,2]]
            prefixes = self.synset_code_to_prefixes(sense_code, pos=pos)
            # [1,2,0,0] -> [1,2,0]
            next_values = sense_code[:len(prefixes)]
            # append zero for full-length sense code (can be detected by len()==n_digits)
            if len(prefixes) > self.n_digits:
                assert len(prefixes) == self.n_digits+1, f"unexpected sense code: {sense_code}"
                next_values += [0]
            for prefix, next_value in zip(prefixes, next_values):
                # [65,1,2] -> "65-1-2"
                lookup_key = self.sequence_to_str(prefix)
                if lookup_key not in result:
                    result_k = {
                        "idx": index,
                        "synset_id": None,
                        "code": None,
                        "next_values": set(),
                        "num_descendents": num_descendents[lookup_key],
                        "is_terminal": True if num_descendents[lookup_key] == 0 else False
                    }
                    _sense_code, pos = self._transform_prefix_to_sense_code_and_pos(prefix)
                    if _sense_code in self:
                        result_k["synset_id"] = self.__getitem__(_sense_code)["id"]
                        result_k["code"] = _sense_code
                        result_k["is_virtual"] = False # this code actually grounded to the synset.
                    else:
                        result_k["is_virtual"] = True # this code is artificial. not grounded to the synset.
                    result[lookup_key] = result_k
                    index += 1

                result[lookup_key]["next_values"].add(next_value)

        # result["1-2"] = {"idx":138, "next_values": {4,1,135,...,}, "num_descendents": 16788}
        return result

    def count_synset_code_prefix_next_values(self, dataset: "WSDTaskDataset", use_index_as_lookup_key: bool = False) -> Dict[str, Dict[int, int]]:
        result = {}
        # initialize by sense code taxonomy.
        for str_prefix, record in self._sense_code_taxonomy.items():
            lookup_key = record["idx"] if use_index_as_lookup_key else str_prefix
            result[lookup_key] = {}
            for possible_value in record["next_values"]:
                result[lookup_key][possible_value] = 0

        # count frequency
        q = progressbar.ProgressBar(max_value=progressbar.UnknownLength)
        q.update(0)
        for idx, record in enumerate(dataset.record_loader()):
            sense_code = record["ground_truth_synset_code"]
            pos = record["pos"]
            prefixes = self.synset_code_to_prefixes(sense_code, pos=pos)
            next_values = sense_code[:len(prefixes)]
            # append zero for full-length sense code (can be detected by len()==n_digits)
            if len(prefixes) > self.n_digits:
                assert len(prefixes) == self.n_digits+1, f"unexpected sense code: {sense_code}"
                next_values += [0]
            for prefix, next_value in zip(prefixes, next_values):
                # ([65,1,2],5) -> ["65-1-2"][5] += 1
                str_prefix = self.sequence_to_str(prefix)
                if use_index_as_lookup_key:
                    lookup_key = self._sense_code_taxonomy[str_prefix]["idx"]
                else:
                    lookup_key = str_prefix
                result[lookup_key][next_value] += 1
            if idx % 10000 == 0:
                q.update(idx)

        return result

    def synset_code_to_prefix(self, partial_sense_code: List[int], pos: Optional[str] = None) -> List[int]:
        n_length = len(partial_sense_code) - partial_sense_code.count(0)
        if pos is not None:
            prefix = [self.pos_index[pos]] + partial_sense_code[:n_length]
        return prefix

    def synset_code_to_prefixes(self, sense_code: List[int], pos: Optional[str] = None) -> List[List[int]]:
        n_length = len(sense_code) - sense_code.count(0)
        if pos is not None:
            lst_prefixes = [[self.pos_index[pos]] + sense_code[:d] for d in range(n_length + 1)]
        return lst_prefixes

    def _get_synset_code_prefix_info(self, sense_code_prefix: List[int]):
        key = self.sequence_to_str(sense_code_prefix)
        return self._sense_code_taxonomy[key]

    def get_synset_code_prefix_info(self, partial_sense_code_or_prefix_index: Union[int, List[int]], pos: Optional[str] = None):
        """
        return the taxonomical information (e.g., next values, number of descendents).
        This is useful for inference including post-hoc

        @rtype: object
        """
        if isinstance(partial_sense_code_or_prefix_index, int):
            lookup_key = self._sense_code_taxonomy_by_index[partial_sense_code_or_prefix_index]
            return self._sense_code_taxonomy[lookup_key]
        else:
            prefix = self.synset_code_to_prefix(partial_sense_code=partial_sense_code_or_prefix_index, pos=pos)
            return self._get_synset_code_prefix_info(sense_code_prefix=prefix)

    def synset_code_to_prefix_ids(self, synset_code: List[int], pos: str, pad: bool = False, trim: bool = False) -> List[int]:
        """
        transform the sense code to the sequence of prefix ids. returned sequence length is non-zero digits + 1.
        This is useful for path-aware embeddings.
        i.e. code=[1,2,0,0], pos="n", n_ary=64 -> [idx("65"), idx("65-1"), idx("65-1-2")]

        @param synset_code: synset code. list of ints.
        @param pos: part-of-speech character such as "n"
        @param pad: pad until the sequence length matches number of digits.
        @param trim: trim if sequence length exceeds the number of digits.
        @return: sequence of prefix ids. list of ints.
        """
        prefixes = self.synset_code_to_prefixes(synset_code, pos=pos)
        lst_ids = [self._get_synset_code_prefix_info(prefix)["idx"] for prefix in prefixes]
        if pad:
            n_pad = len(synset_code) - len(lst_ids)
            if n_pad > 0:
                # repeat last element until length matches with input.
                lst_ids += [lst_ids[-1]] * n_pad
        if trim:
            # trim tail which exceeds input length.
            lst_ids = lst_ids[:len(synset_code)]
        return lst_ids

    def __getitem__(self, synset_id_or_synset_code: Union[str, List[int]]):
        if isinstance(synset_id_or_synset_code, str):
            return self._synsets[synset_id_or_synset_code]
        elif isinstance(synset_id_or_synset_code, list):
            key = self.sequence_to_str(synset_id_or_synset_code)
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
        if synset_id in self:
            return self[synset_id]["code"]
        else:
            return None

    @property
    def n_digits(self):
        if not hasattr(self, "_n_digits"):
            def _apply_function(it_codes):
                return max([len(code) for code in it_codes])
            self._n_digits = self._apply(apply_field_name="code", disable_transform_functions=False,
                                         apply_function=_apply_function)
        return self._n_digits

    @property
    def n_ary(self):
        if not hasattr(self, "_n_ary"):
            def _apply_function(it_codes):
                return max([max(code) for code in it_codes]) + 1
            self._n_ary = self._apply(apply_field_name="code", disable_transform_functions=False,
                                      apply_function=_apply_function)
        return self._n_ary

    @property
    def n_synset(self):
        return len(self._synsets) // 2

    @property
    def n_synset_code_prefix(self):
        # synset code接頭辞の異なり数．n_synset以上の値になる．
        # n_synsetとは必ずしも一致しない．codingの際にひとつの分岐を2桁以上で表現しうるためである．
        return len(self._sense_code_taxonomy)

    @property
    def pos_tagset(self) -> List[str]:
        """
        set of part-of-speech tags in the dataset. e.g., ["n","v"]

        @return: set of part-of-speech tags
        """
        return self.distinct_values(column="pos")

    @property
    def pos_index(self) -> Dict[str, int]:
        """
        part-of-speech tag to index which starts at n_ary. e.g., {"n":64, "v":65}

        @return:
        """
        if not hasattr(self, "_pos_index"):
            pos_index = {}
            for idx, pos in enumerate(sorted(list(self.pos_tagset))):
                pos_index[pos] = self.n_ary + idx
            self._pos_index = pos_index
            self._pos_index_rev = {idx:pos for pos,idx in pos_index.items()}
        return self._pos_index

    @property
    def verbose(self):
        v = super().verbose
        v["n_digits"] = self.n_digits
        v["n_ary"] = self.n_ary
        v["n_synset"] = self.n_synset
        v["n_synset_code_prefix"] = self.n_synset_code_prefix
        return v

    @property
    def sense_code_prefix_index(self):
        prefix_index = {prefix:record["idx"] for prefix, record in self._sense_code_taxonomy.items()}
        return prefix_index