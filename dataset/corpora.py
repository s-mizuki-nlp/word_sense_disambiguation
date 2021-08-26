#!/usr/bin/env python
# -*- coding:utf-8 -*-

import io, os, json
from typing import Union, Collection, Optional, Dict, Any, Iterable, Callable, List
from torch.utils.data import IterableDataset

from dataset_preprocessor import utils
from ._base import AbstractFormatDataset

class NDJSONDataset(AbstractFormatDataset, IterableDataset):

    def __init__(self, path: str,
                 binary: bool,
                 transform_functions=None,
                 filter_function: Optional[Union[Callable, List[Callable]]] = None,
                 n_rows: Optional[int] = None,
                 description: str = "",
                 **kwargs_for_json_loads):
        """
        NDJSONフォーマットを読み込むクラス

        :param path: NDJSONファイルのパス
        :param binary: レコードがpickle化されているか否か
        :param n_rows: 読み込む最大レコード数
        :param transform_functions: データ変形定義，Dictionaryを指定．keyはフィールド名，valueは変形用関数
        :param filter_function: 除外するか否かを判定する関数
        :param description: 説明
        :param kwargs_for_json_loads: `json.loads()` methodに渡すキーワード引数．binary=Trueの場合は無視．
        """

        super().__init__(path, transform_functions, filter_function, n_rows, description)
        self._binary = binary
        self._kwargs_for_json_loads = kwargs_for_json_loads

    def _record_loader_text(self):
        iter_records = io.open(self._path, mode="r")
        for record in iter_records:
            obj = json.loads(record, **self._kwargs_for_json_loads)
            yield obj

    def _record_loader_binary(self):
        iter_records = utils.iter_read_pickled_object(self._path)
        for record in iter_records:
            yield record

    def _record_loader(self):
        if self._binary:
            return self._record_loader_binary()
        else:
            return self._record_loader_text()

    @property
    def verbose(self):
        ret = {
            "path": self._path,
            "nrows": self.__len__(),
            "description": self._description,
            "filter_function": self._filter_function,
            "transform_functions": self._transform_functions
        }
        return ret
