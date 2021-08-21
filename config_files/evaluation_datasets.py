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
word_sense_disambiguation = {
    "ALL": {
        "path_corpus": os.path.join(DIR_EVALSET, "ALL/ALL.data.xml"),
        "path_ground_truth_labels": os.path.join(DIR_EVALSET, "ALL/ALL.gold.key.txt"),
        "lookup_candidate_senses": True,
        "description": "WSD dataset: ALL",
    }
}
