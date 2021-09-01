#!/usr/bin/env python
# -*- coding:utf-8 -*-

from typing import Optional, Union, Callable, List, Any, Dict
import json
import h5py

from torch.utils.data import IterableDataset
from ._base import AbstractFormatDataset
from .encoder import convert_compressed_format_to_list_of_tensors


class BERTEmbeddingsBatchDataset(AbstractFormatDataset, IterableDataset):

    def __init__(self,
                 path: str,
                 transform_functions=None,
                 filter_function: Optional[Union[Callable, List[Callable]]] = None,
                 n_rows: Optional[int] = None,
                 description: str = ""):

        super().__init__(path, transform_functions, filter_function, n_rows, description)

    def _record_loader(self):
        ifs = h5py.File(self._path, mode="r")

        for group_name, group in ifs.items():
            record = {
                "embeddings": group["embeddings"][()],
                "sequence_lengths": group["sequence_lengths"][()],
                "records": json.loads(group["records"][()])
            }
            yield record

        ifs.close()


class BERTEmbeddingsDataset(BERTEmbeddingsBatchDataset):

    def __init__(self,
                 path: str,
                 padding: bool = False,
                 max_sequence_length: Optional[int] = None,
                 transform_functions=None,
                 filter_function: Optional[Union[Callable, List[Callable]]] = None,
                 n_rows: Optional[int] = None,
                 description: str = ""):

        super().__init__(path, transform_functions, filter_function, n_rows, description)
        self._padding = padding
        self._max_sequence_length = max_sequence_length

    def _record_loader(self):

        iter_batch = super()._record_loader()

        for batch in iter_batch:
            lst_embeddings = convert_compressed_format_to_list_of_tensors(embeddings=batch["embeddings"],
                                                                 lst_sequence_lengths=batch["sequence_lengths"],
                                                                 padding=self._padding,
                                                                 max_sequence_length=self._max_sequence_length)
            lst_records = batch["records"]

            iter_records = zip(lst_embeddings, batch["sequence_lengths"], lst_records)
            for embedding, seq_len, record in iter_records:
                dict_record = {
                    "embedding": embedding,
                    "sequence_length": seq_len,
                    "record": record
                }
                yield dict_record

    @property
    def n_dim(self):
        record = next(iter(self))
        t = record["embedding"]
        return t.shape[-1]
