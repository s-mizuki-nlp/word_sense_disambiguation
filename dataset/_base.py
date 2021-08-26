#!/usr/bin/env python
# -*- coding:utf-8 -*-
import io
import json
import os
from typing import Optional, Union, Callable, List, Any, Dict
from abc import ABCMeta, abstractmethod

from torch.utils.data import IterableDataset

from dataset_preprocessor import utils


class AbstractFormatDataset(metaclass=ABCMeta):

    def __init__(self, path: str,
                 transform_functions=None,
                 filter_function: Optional[Union[Callable, List[Callable]]] = None,
                 n_rows: Optional[int] = None,
                 description: str = ""):
        """
        様々なフォーマットのデータセットを読み込む基底クラス

        :param path: NDJSONファイルのパス
        :param n_rows: 読み込む最大レコード数
        :param transform_functions: データ変形定義，Dictionaryを指定．keyはフィールド名，valueは変形用関数
        :param filter_function: 除外するか否かを判定する関数
        :param description: 説明
        """

        super().__init__()
        self._path = path

        assert os.path.exists(path), f"invalid path specified: {path}"

        self._description = description
        self._transform_functions = transform_functions

        if filter_function is None:
            self._filter_function = []
        elif isinstance(filter_function, list):
            self._filter_function = filter_function
        elif not isinstance(filter_function, list):
            self._filter_function = [filter_function]

        self._n_rows = n_rows
        self._n_sample = None

    @abstractmethod
    def _record_loader(self):
        pass

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

    def __iter__(self):
        if isinstance(self._transform_functions, dict):
            for field_name, function in self._transform_functions.items():
                if hasattr(function, "reset"):
                    function.reset()

        iter_records = self._record_loader()
        n_read = 0
        for record in iter_records:
            # transform each field of the entry
            entry = self._transform(record)

            # verify the entry is valid or not
            if self._filter(entry) == True:
                continue

            yield entry

            n_read += 1
            if self._n_rows is not None:
                if n_read >= self._n_rows:
                    break

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