#!/usr/bin/env python
# -*- coding:utf-8 -*-

from typing import Dict, Any
from ._supervised_base import WSDTaskEvaluatorBase


# practical implementation for supervised task evaluation #

class ToyWSDTaskEvaluator(WSDTaskEvaluatorBase):

    def predict(self, input: Dict[str, Any]):
        return ["art%1:09:00::"]