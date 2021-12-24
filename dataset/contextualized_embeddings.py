#!/usr/bin/env python
# -*- coding:utf-8 -*-

from typing import Optional, Union, Callable, List, Any, Dict
import json, math
import h5py

from torch.utils.data import IterableDataset, get_worker_info
from ._base import AbstractFormatDataset
from .encoder import convert_compressed_format_to_list_of_tensors


class BERTEmbeddingsBatchDataset(AbstractFormatDataset, IterableDataset):

    def __init__(self,
                 path: str,
                 transform_functions=None,
                 filter_function: Optional[Union[Callable, List[Callable]]] = None,
                 n_rows: Optional[int] = None,
                 return_record_only: bool = False,
                 description: str = ""):

        super().__init__(path, transform_functions, filter_function, n_rows, description)

        # save statistics
        ifs = h5py.File(self._path, mode="r")
        group_names = []
        num_sentences = 0
        for group_name, group in ifs.items():
            n_seq = group["sequence_lengths"].shape[0]
            num_sentences += n_seq
            group_names.append(group_name)
        self._group_names = group_names
        self._num_groups = len(group_names)
        self._num_sentences = num_sentences
        self._n_dim = group["embeddings"].shape[-1]
        self._return_record_only = return_record_only

        ifs.close()

    def _record_loader(self, start: int = None, end: int = None):
        start = 0 if start is None else start
        end = self.num_groups if end is None else end

        lst_group_names = self._group_names[start:end]
        ifs = h5py.File(self._path, mode="r")
        for group_name in lst_group_names:
            group = ifs.get(group_name)

            record = {
                "records": json.loads(group["records"][()])
            }
            if not self._return_record_only:
                record["embeddings"] = group["embeddings"][()]
                record["sequence_lengths"] = group["sequence_lengths"][()]

            yield record

        ifs.close()

    @property
    def num_groups(self):
        return self._num_groups

    @property
    def num_sentences(self):
        return self._num_sentences

    @property
    def n_dim(self):
        return self._n_dim


class BERTEmbeddingsDataset(BERTEmbeddingsBatchDataset):

    def __init__(self,
                 path: str,
                 padding: bool = False,
                 max_sequence_length: Optional[int] = None,
                 transform_functions=None,
                 filter_function: Optional[Union[Callable, List[Callable]]] = None,
                 n_rows: Optional[int] = None,
                 return_record_only: bool = False,
                 description: str = ""):

        super().__init__(path, transform_functions, filter_function, n_rows, return_record_only, description)
        self._padding = padding
        self._max_sequence_length = max_sequence_length

    def _record_loader(self):
        worker_info = get_worker_info()
        if worker_info is None: # single-process data loading, return the full iterator
            start = end = None
        else: # in a worker process
            # split workload
            per_worker = int(math.ceil(self.num_groups / float(worker_info.num_workers)))
            worker_id = worker_info.id
            start = 0 + worker_id * per_worker
            end = min(start + per_worker, self.num_groups)

        iter_batch = super()._record_loader(start=start, end=end)
        for batch in iter_batch:
            lst_records = batch["records"]

            if self._return_record_only:
                for record in lst_records:
                    dict_record = {
                        "record": record
                    }
                    yield dict_record
            else:
                lst_sequence_lengths = batch["sequence_lengths"]
                lst_embeddings = convert_compressed_format_to_list_of_tensors(embeddings=batch["embeddings"],
                                                                 lst_sequence_lengths=lst_sequence_lengths,
                                                                 padding=self._padding,
                                                                 max_sequence_length=self._max_sequence_length)
                iter_records = zip(lst_embeddings, lst_sequence_lengths, lst_records)
                for embedding, seq_len, record in iter_records:
                    dict_record = {
                        "embedding": embedding,
                        "sequence_length": seq_len,
                        "record": record
                    }
                    yield dict_record

    @property
    def return_record_only(self):
        return self._return_record_only

    @return_record_only.setter
    def return_record_only(self, flag: bool):
        self._return_record_only = flag

    @property
    def verbose(self):
        ret = {
            "path": self._path,
            "num_sentences":self.num_sentences,
            "n_dim":self.n_dim,
            "return_record_only":self.return_record_only,
            "description":self._description
        }
        return ret