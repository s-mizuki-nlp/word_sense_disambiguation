#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import os
from dataset.filter import EmptyFilter, DictionaryFilter
_no_entity_sentence_filter = EmptyFilter(check_field_names=["entities"])
_noun_verb_entity_selector = DictionaryFilter(includes={"pos":{"n","v"}})

from .utils import pick_first_available_path

DIR_WORDNET_GLOSS = "/home/sakae/Windows/dataset/word_sense_disambiguation/wordnet_gloss_corpus/"
DIR_WORDNET_GLOSS_EMBEDDINGS = "/home/sakae/Windows/dataset/word_sense_disambiguation/wordnet_gloss_corpus/bert_embeddings/"
DIR_EXT_WORDNET_GLOSS = "/home/sakae/Windows/dataset/word_sense_disambiguation/wordnet_gloss_augmentation/"
DIR_EXT_WORDNET_GLOSS_EMBEDDINGS = "/home/sakae/Windows/dataset/word_sense_disambiguation/wordnet_gloss_augmentation/bert_embeddings/"
DIR_TRAIN_UNSUPERVISED_LOCAL = "/tmp/sakae/"

cfg_training = {
    "WordNet-Gloss-noun-verb": {
        "lst_path_gloss_corpus": [
            os.path.join(DIR_WORDNET_GLOSS, "WordNet-3.0/glosstag/merged/noun.xml"),
            os.path.join(DIR_WORDNET_GLOSS, "WordNet-3.0/glosstag/merged/verb.xml")
        ],
        "concat_hypernym_definition_sentence": False,
        "description": "Automatically annotated WordNet Gloss corpus for noun and verb."
    },
    "WordNet-Gloss-Augmented-noun-verb": {
        "lst_path_gloss_corpus": [
            os.path.join(DIR_WORDNET_GLOSS, "WordNet-3.0/glosstag/merged/noun.xml"),
            os.path.join(DIR_WORDNET_GLOSS, "WordNet-3.0/glosstag/merged/verb.xml")
        ],
        "concat_hypernym_definition_sentence": True,
        "description": "Automatically annotated WordNet Gloss corpus augmented using the definition sentence of hypernym. noun and verb."
    },
    "Extended-WordNet-Gloss-noun-verb": {
        "lst_path_gloss_corpus": [
            os.path.join(DIR_EXT_WORDNET_GLOSS, "sentence_dict_n"),
            os.path.join(DIR_EXT_WORDNET_GLOSS, "sentence_dict_v")
        ],
        "description": "WordNet Gloss Corpus which is extended using Baidu translation. Used in [Wang and Wang, EMNLP2020]. for noun and verb."
    },
    "WordNet-Gloss-noun-verb-bert-large-cased": {
        "path": pick_first_available_path(
            os.path.join(DIR_TRAIN_UNSUPERVISED_LOCAL, "bert-large-cased_WordNet-Gloss-noun-verb.hdf5"),
            os.path.join(DIR_WORDNET_GLOSS_EMBEDDINGS, "bert-large-cased_WordNet-Gloss-noun-verb.hdf5")
        ),
        "padding": False,
        "max_sequence_length": None,
        "description": "WordNet Gloss corpora encoded by BERT-large-cased."
    },
    "WordNet-Gloss-Augmented-noun-verb-bert-large-cased": {
        "path": pick_first_available_path(
            os.path.join(DIR_TRAIN_UNSUPERVISED_LOCAL, "bert-large-cased_WordNet-Gloss-Augmented-noun-verb.hdf5"),
            os.path.join(DIR_WORDNET_GLOSS_EMBEDDINGS, "bert-large-cased_WordNet-Gloss-Augmented-noun-verb.hdf5")
        ),
        "padding": False,
        "max_sequence_length": None,
        "description": "WordNet Gloss augmented using hypernym def. sentences corpora encoded by BERT-large-cased."
    },
    "Extended-WordNet-Gloss-noun-verb-bert-large-cased": {
        "path": pick_first_available_path(
            os.path.join(DIR_TRAIN_UNSUPERVISED_LOCAL, "bert-large-cased_Extended-WordNet-Gloss-noun-verb.hdf5"),
            os.path.join(DIR_EXT_WORDNET_GLOSS_EMBEDDINGS, "bert-large-cased_Extended-WordNet-Gloss-noun-verb.hdf5")
        ),
        "padding": False,
        "max_sequence_length": None,
        "description": "Extended WordNet Gloss corpora encoded by BERT-large-cased."
    },
}