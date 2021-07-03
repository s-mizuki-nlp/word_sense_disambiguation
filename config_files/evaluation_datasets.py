#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import os
from dataset.transform import FieldTypeConverter

DIR_EVALSET = "/home/sakae/Windows/dataset/word_sense_disambiguation/WSD_Evaluation_Framework/Evaluation_Datasets/"

# evaluation dataset for all-words WSD task
cfg_evaluation_datasets_word_sense_disambiguation = {
    "ALL": {
        "path_data": os.path.join(DIR_EVALSET, "ALL/ALL.data.xml"),
        "path_gold": os.path.join(DIR_EVALSET, "ALL/ALL.gold.key.txt"),
        "description": "WSD dataset: ALL",
    }
}
