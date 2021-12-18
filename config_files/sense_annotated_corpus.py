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

DIR_EVALSET = "/home/sakae/Windows/dataset/word_sense_disambiguation/WSD_Evaluation_Framework/Evaluation_Datasets/"
DIR_EVALSET_EMBEDDINGS = "/home/sakae/Windows/dataset/word_sense_disambiguation/WSD_Evaluation_Framework/bert_embeddings/"
DIR_TRAINSET = "/home/sakae/Windows/dataset/word_sense_disambiguation/WSD_Training_Corpora/"
DIR_TRAINSET_EMBEDDINGS = "/home/sakae/Windows/dataset/word_sense_disambiguation/WSD_Training_Corpora/bert_embeddings/"
DIR_WORDNET_GLOSS = "/home/sakae/Windows/dataset/word_sense_disambiguation/wordnet_gloss_corpus/"
DIR_WORDNET_GLOSS_EMBEDDINGS = "/home/sakae/Windows/dataset/word_sense_disambiguation/wordnet_gloss_corpus/bert_embeddings/"
DIR_EXT_WORDNET_GLOSS = "/home/sakae/Windows/dataset/word_sense_disambiguation/wordnet_gloss_augmentation/"
DIR_EXT_WORDNET_GLOSS_EMBEDDINGS = "/home/sakae/Windows/dataset/word_sense_disambiguation/wordnet_gloss_augmentation/bert_embeddings/"

# evaluation dataset for all-words WSD task
cfg_evaluation = {
    "WSDEval-ALL": {
        "path_corpus": os.path.join(DIR_EVALSET, "ALL/ALL.data.xml"),
        "path_ground_truth_labels": os.path.join(DIR_EVALSET, "ALL/ALL.gold.key.txt"),
        "lookup_candidate_senses": True,
        "description": "WSD Evaluation Framework dataset [Raganato+, 2017]: ALL",
    },
    "WSDEval-noun-verb": {
        "path_corpus": os.path.join(DIR_EVALSET, "ALL/ALL.data.xml"),
        "path_ground_truth_labels": os.path.join(DIR_EVALSET, "ALL/ALL.gold.key.txt"),
        "lookup_candidate_senses": True,
        "entity_filter_function": _noun_verb_entity_selector,
        "description": "WSD Evaluation Framework dataset [Raganato+, 2017]: Noun and verb entity only.",
    },
    "WSDEval-ALL-bert-large-cased": {
        "path":os.path.join(DIR_EVALSET_EMBEDDINGS, "bert-large-cased_wsdeval-all.hdf5"),
        "padding": False,
        "max_sequence_length": None,
        "filter_function":None,
        "description": "WSD Evaluation Framework dataset [Raganato+, 2017] encoded by BERT-large-cased."
    },
    "WSDEval-noun-verb-bert-large-cased": {
        "path":os.path.join(DIR_EVALSET_EMBEDDINGS, "bert-large-cased_WSDEval-noun-verb.hdf5"),
        "padding": False,
        "max_sequence_length": None,
        "filter_function":None,
        "description": "WSD Evaluation Framework dataset [Raganato+, 2017]: Noun and verb entity only, encoded by BERT-large-cased."
    }
}


cfg_training = {
    "SemCor": {
        "path_corpus": os.path.join(DIR_TRAINSET, "SemCor/semcor.data.xml"),
        "path_ground_truth_labels": os.path.join(DIR_TRAINSET, "SemCor/semcor.gold.key.txt"),
        "lookup_candidate_senses": True,
        "filter_function": _no_entity_sentence_filter,
        "description": "WSD SemCor corpora, excluding no-sense-annotated sentences.",
    },
    "SemCor-noun-verb": {
        "path_corpus": os.path.join(DIR_TRAINSET, "SemCor/semcor.data.xml"),
        "path_ground_truth_labels": os.path.join(DIR_TRAINSET, "SemCor/semcor.gold.key.txt"),
        "lookup_candidate_senses": True,
        "filter_function": _no_entity_sentence_filter,
        "entity_filter_function": _noun_verb_entity_selector,
        "description": "WSD SemCor corpora, excluding no-sense-annotated sentences. selects noun and verb entity only.",
    },
    "SemCor-bert-large-cased": {
        "path":os.path.join(DIR_TRAINSET_EMBEDDINGS, "bert-large-cased_SemCor.hdf5"),
        "padding": False,
        "max_sequence_length": None,
        "description": "WSD SemCor corpora (excluding no-sense-annotated sentences) encoded by BERT-large-cased."
    },
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
            os.path.join(DIR_EXT_WORDNET_GLOSS, "sentence_dict_n")
        ],
        "description": "WordNet Gloss Corpus which is extended using Baidu translation. Used in [Wang and Wang, EMNLP2020]. for noun and verb."
    },
}