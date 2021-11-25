#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import os
from dataset.transform import top_digit_remover, top_digits_remover
from dataset.filter import DictionaryFilter
_root_synset_filter = DictionaryFilter(excludes={"id":{"entity.n.01", "verb_dummy_root.v.01"}})
_noun_selector = DictionaryFilter(includes={"pos":{"n",}})

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
    "WordNet-noun-verb-incl-instance-random-assignment": {
        "path": os.path.join(DIR_LEXICAL_KNOWLEDGE, "lemma_dictionary_pos-n+v_ary-64_assign-random_incl-instance-of.jsonl"),
        "binary": False,
        "monosemous_entity_only": False,
        "lemma_lowercase": True,
        "description": "WordNet(N+V), includes instance-of, N_ary = 64. value assignment rule: random",
    },
    "WordNet-noun-incl-instance-random-assignment": {
        "path": os.path.join(DIR_LEXICAL_KNOWLEDGE, "lemma_dictionary_pos-n+v_ary-64_assign-random_incl-instance-of.jsonl"),
        "filter_function": _noun_selector,
        "binary": False,
        "monosemous_entity_only": False,
        "lemma_lowercase": True,
        "description": "WordNet(N), includes instance-of, N_ary = 64. value assignment rule: random",
    },
    "WordNet-noun-verb-incl-instance-unlimited-ary-random-assignment": {
        "path": os.path.join(DIR_LEXICAL_KNOWLEDGE, "lemma_dictionary_pos-n+v_ary-None_assign-random_incl-instance-of.jsonl"),
        "binary": False,
        "monosemous_entity_only": False,
        "lemma_lowercase": True,
        "description": "WordNet(N+V), includes instance-of, N_ary = unlimited. value assignment rule: random",
    },
    # "WordNet-noun-verb-incl-instance-without-top-digit": {
    #     "path": os.path.join(DIR_LEXICAL_KNOWLEDGE, "lemma_dictionary_pos-n+v_ary-64_incl-instance-of.jsonl"),
    #     "transform_functions": {"synset_codes": top_digits_remover},
    #     "binary": False,
    #     "lemma_lowercase": True,
    #     "description": "WordNet(N+V), includes instance-of, N_ary = 64, Trim most significant (=top) digit.",
    # },
    "WordNet-noun-verb-incl-instance-monosemous": {
        "path": os.path.join(DIR_LEXICAL_KNOWLEDGE, "lemma_dictionary_pos-n+v_ary-64_incl-instance-of.jsonl"),
        "binary": False,
        "monosemous_entity_only": True,
        "lemma_lowercase": True,
        "description": "WordNet(N+V), includes instance-of, monosemous only, N_ary = 64.",
    },
    "WordNet-noun-verb-random-assignment": {
        "path": os.path.join(DIR_LEXICAL_KNOWLEDGE, "lemma_dictionary_pos-n+v_ary-64_assign-random.jsonl"),
        "binary": False,
        "monosemous_entity_only": False,
        "lemma_lowercase": True,
        "description": "WordNet(N+V), excludes instance-of lemmas, N_ary = 64. value assignment rule: random",
    },
    "WordNet-noun-verb-incl-instance-unlimited-ary": {
        "path": os.path.join(DIR_LEXICAL_KNOWLEDGE, "lemma_dictionary_pos-n+v_ary-None_incl-instance-of.jsonl"),
        "binary": False,
        "monosemous_entity_only": False,
        "lemma_lowercase": True,
        "description": "WordNet(N+V), includes instance-of, N_ary = unlimited.",
    },
    "WordNet-noun-verb-unlimited-ary-random-assignment": {
        "path": os.path.join(DIR_LEXICAL_KNOWLEDGE, "lemma_dictionary_pos-n+v_ary-None_assign-random.jsonl"),
        "binary": False,
        "monosemous_entity_only": False,
        "lemma_lowercase": True,
        "description": "WordNet(N+V), excludes instance-of lemmas, N_ary = unlimited. value assignment rule: random",
    }
}

## synset(=word sense) datasets
cfg_synset_datasets = {
    "WordNet-noun-verb-incl-instance": {
        "path": os.path.join(DIR_LEXICAL_KNOWLEDGE, "synset_taxonomy_pos-n+v_ary-64_incl-instance-of.jsonl"),
        "binary": False,
        "lemma_lowercase": True,
        "description": "DEFAULT Dataset. WordNet(N+V), includes instance-of, N_ary = 64.",
    },
    "WordNet-noun-verb-incl-instance-random-assignment": {
        "path": os.path.join(DIR_LEXICAL_KNOWLEDGE, "synset_taxonomy_pos-n+v_ary-64_assign-random_incl-instance-of.jsonl"),
        "binary": False,
        "lemma_lowercase": True,
        "description": "WordNet(N+V), includes instance-of, N_ary = 64. value assignment rule: random",
    },
    "WordNet-noun-incl-instance-random-assignment": {
        "path": os.path.join(DIR_LEXICAL_KNOWLEDGE, "synset_taxonomy_pos-n+v_ary-64_assign-random_incl-instance-of.jsonl"),
        "filter_function":_noun_selector,
        "binary": False,
        "lemma_lowercase": True,
        "description": "WordNet(N), includes instance-of, N_ary = 64. value assignment rule: random",
    },
    "WordNet-noun-verb-incl-instance-unlimited-ary-random-assignment": {
        "path": os.path.join(DIR_LEXICAL_KNOWLEDGE, "synset_taxonomy_pos-n+v_ary-None_assign-random_incl-instance-of.jsonl"),
        "binary": False,
        "lemma_lowercase": True,
        "description": "WordNet(N+V), includes instance-of, N_ary = unlimited. value assignment rule: random",
    },
    # "WordNet-noun-verb-incl-instance-without-top-digit": {
    #     "path": os.path.join(DIR_LEXICAL_KNOWLEDGE, "synset_taxonomy_pos-n+v_ary-64_incl-instance-of.jsonl"),
    #     "transform_functions": {"code": top_digit_remover},
    #     "filter_function": _root_synset_filter,
    #     "binary": False,
    #     "lemma_lowercase": True,
    #     "description": "WordNet(N+V), includes instance-of, N_ary = 64, Trim most significant (=top) digit and removed root entity (i.e. entity.n.01)",
    # },
    "WordNet-noun-verb-random-assignment": {
        "path": os.path.join(DIR_LEXICAL_KNOWLEDGE, "synset_taxonomy_pos-n+v_ary-64_assign-random.jsonl"),
        "binary": False,
        "lemma_lowercase": True,
        "description": "WordNet(N+V), excludes instance-of lemmas, N_ary = 64. value assignment rule: random",
    },
    "WordNet-noun-verb-incl-instance-unlimited-ary": {
        "path": os.path.join(DIR_LEXICAL_KNOWLEDGE, "synset_taxonomy_pos-n+v_ary-None_incl-instance-of.jsonl"),
        "binary": False,
        "lemma_lowercase": True,
        "description": "WordNet(N+V), includes instance-of, N_ary = unlimited.",
    },
    "WordNet-noun-verb-unlimited-ary-random-assignment": {
        "path": os.path.join(DIR_LEXICAL_KNOWLEDGE, "synset_taxonomy_pos-n+v_ary-None_assign-random.jsonl"),
        "binary": False,
        "lemma_lowercase": True,
        "description": "WordNet(N+V), excludes instance-of lemmas, N_ary = unlimited. value assignment rule: random",
    },
}

