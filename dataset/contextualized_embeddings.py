#!/usr/bin/env python
# -*- coding:utf-8 -*-

from typing import Optional, Union, Callable, List, Any, Dict
import json
import h5py

from ._base import AbstractFormatDataset
from .encoder import convert_compressed_format_to_batch_format


class BERTEmbeddingsBatchDataset(AbstractFormatDataset):

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
                 max_sequence_length: Optional[int] = None,
                 transform_functions=None,
                 filter_function: Optional[Union[Callable, List[Callable]]] = None,
                 n_rows: Optional[int] = None,
                 description: str = ""):

        super().__init__(path, transform_functions, filter_function, n_rows, description)
        self._max_sequence_length = max_sequence_length

    def _record_loader(self):

        iter_batch = super()._record_loader()

        for batch in iter_batch:
            # embeddings: (n_batch, max_seq_len, n_dim)
            dict_ = convert_compressed_format_to_batch_format(embeddings=batch["embeddings"],
                                                                   lst_sequence_lengths=batch["sequence_lengths"],
                                                                   max_sequence_length=self._max_sequence_length,
                                                                   return_tensor=False)
            embeddings = dict_["embeddings"]
            attention_masks = dict_["attention_mask"]
            lst_records = batch["records"]

            iter_records = zip(embeddings, batch["sequence_lengths"], attention_masks, lst_records)
            for embedding, seq_len, attention_mask, record in iter_records:
                dict_record = {
                    "embedding": embedding,
                    "sequence_length": seq_len,
                    "attention_mask": attention_mask,
                    "record": record
                }
                yield dict_record


