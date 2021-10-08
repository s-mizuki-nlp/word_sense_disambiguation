#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import os
import numpy as np
from dataset.transform import FieldTypeConverter
_distance_str_to_float = FieldTypeConverter(dict_field_type_converter={"distance":np.float32})

DIR_LEXICAL_KNOWLEDGE = "/home/sakae/Windows/dataset/word_sense_disambiguation/wordnet_taxonomy/"

# lexical knowledge: sense-code annotated lemmas and synsets extracted from WordNet.

## lemma datasets
cfg_lemma_datasets = {
    "WordNet-noun-verb-incl-instance": {
        "path": os.path.join(DIR_LEXICAL_KNOWLEDGE, "lemma_dictionary_pos-n+v_ary-64_incl-instance-of.jsonl"),
        "binary": False,
        "monosemous_entity_only": False,
        "lemma_lowercase": True,
        "description": "DEFAULT Dataset. WordNet(N+V), includes instance-of, N_ary = 64.",
    },
    "WordNet-noun-verb-incl-instance-monosemous": {
        "path": os.path.join(DIR_LEXICAL_KNOWLEDGE, "lemma_dictionary_pos-n+v_ary-64_incl-instance-of.jsonl"),
        "binary": False,
        "monosemous_entity_only": True,
        "lemma_lowercase": True,
        "description": "DEFAULT Dataset. WordNet(N+V), includes instance-of, monosemous only, N_ary = 64.",
    },
    "WordNet-noun-verb": {
        "path": os.path.join(DIR_LEXICAL_KNOWLEDGE, "lemma_dictionary_pos-n+v_ary-64.jsonl"),
        "binary": False,
        "monosemous_entity_only": False,
        "lemma_lowercase": True,
        "description": "WordNet(N+V), excludes instance-of, N_ary = 64.",
    },
    "WordNet-noun-verb-incl-instance-unlimited-ary": {
        "path": os.path.join(DIR_LEXICAL_KNOWLEDGE, "lemma_dictionary_pos-n+v_ary-None_incl-instance-of.jsonl"),
        "binary": False,
        "monosemous_entity_only": False,
        "lemma_lowercase": True,
        "description": "WordNet(N+V), includes instance-of, N_ary = unlimited.",
    },
    "WordNet-noun-verb-unlimited-ary": {
        "path": os.path.join(DIR_LEXICAL_KNOWLEDGE, "lemma_dictionary_pos-n+v_ary-None.jsonl"),
        "binary": False,
        "monosemous_entity_only": False,
        "lemma_lowercase": True,
        "description": "WordNet(N+V), excludes instance-of, N_ary = unlimited.",
    }
}

## synset(=word sense) datasets
cfg_synset_datasets = {
    "WordNet-noun-verb-incl-instance": {
        "path": os.path.join(DIR_LEXICAL_KNOWLEDGE, "synset_taxonomy_pos-n+v_ary-64_incl-instance-of.jsonl"),
        "binary": False,
        "lemma_lowercase": True,
        "lookup_lemma_keys": True,
        "description": "DEFAULT Dataset. WordNet(N+V), includes instance-of, N_ary = 64.",
    },
    "WordNet-noun-verb": {
        "path": os.path.join(DIR_LEXICAL_KNOWLEDGE, "synset_taxonomy_pos-n+v_ary-64.jsonl"),
        "binary": False,
        "lemma_lowercase": True,
        "lookup_lemma_keys": True,
        "description": "WordNet(N+V), excludes instance-of, N_ary = 64.",
    },
    "WordNet-noun-verb-incl-instance-unlimited-ary": {
        "path": os.path.join(DIR_LEXICAL_KNOWLEDGE, "synset_taxonomy_pos-n+v_ary-None_incl-instance-of.jsonl"),
        "binary": False,
        "lemma_lowercase": True,
        "lookup_lemma_keys": True,
        "description": "WordNet(N+V), includes instance-of, N_ary = unlimited.",
    },
    "WordNet-noun-verb-unlimited-ary": {
        "path": os.path.join(DIR_LEXICAL_KNOWLEDGE, "synset_taxonomy_pos-n+v_ary-None.jsonl"),
        "binary": False,
        "lemma_lowercase": True,
        "lookup_lemma_keys": True,
        "description": "WordNet(N+V), excludes instance-of, N_ary = unlimited.",
    },
}

