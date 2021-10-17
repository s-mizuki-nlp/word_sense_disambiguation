#!/usr/bin/env python
# -*- coding:utf-8 -*-

from typing import Dict, Any, Iterable
from ._supervised_base import WSDTaskEvaluatorBase


# practical implementation for supervised task evaluation #

class ToyWSDTaskEvaluator(WSDTaskEvaluatorBase):

    def predict(self, input: Dict[str, Any]):
        return ["art%1:09:00::"]


class MostFrequentSenseWDSTaskEvaluator(WSDTaskEvaluatorBase):

    def predict(self, input: Dict[str, Any]) -> Iterable[str]:
        # ToDo: implement most freuqent sense method.
        pass