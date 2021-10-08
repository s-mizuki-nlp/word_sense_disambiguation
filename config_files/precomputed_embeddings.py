#!/usr/bin/env python
# -*- coding:utf-8 -*-

import sys, io, os

DIR_TRAIN_UNSUPERVISED = "/home/sakae/Windows/dataset/word_sense_disambiguation/monosemous_word_annotated_corpus/bert_embeddings/"
DIR_TRAIN = "/home/sakae/Windows/dataset/word_sense_disambiguation/WSD_Training_Corpora/bert_embeddings/"
DIR_EVAL = "/home/sakae/Windows/dataset/word_sense_disambiguation/WSD_Evaluation_Framework/bert_embeddings/"

cfg_evaluation = {
    "WSDEval-all": {
        "path":os.path.join(DIR_EVAL, "bert-large-cased_wsdeval-all.hdf5"),
        "padding": False,
        "max_sequence_length": None,
        "filter_function":None,
        "description": "BERT-large-cased. WSD Evaluation Framework dataset [Raganato+, 2017]"
    }
}

cfg_training = {
    "wikitext103": {
        "path":os.path.join(DIR_TRAIN_UNSUPERVISED, "bert-base-cased_wikitext103_train_freq=10-100_len=6-128.hdf5"),
        "padding": False,
        "max_sequence_length": None,
        "description": "BERT-base-cased. Wikitext-103 trainset, freq=10~100, length=6~128."
    },
    "wiki40b-first-paragraph": {
        "path":os.path.join(DIR_TRAIN_UNSUPERVISED, "bert-base-cased_wiki40b-train-first-paragraph_freq=10-100_len=6-128.hdf5"),
        "padding": False,
        "max_sequence_length": None,
        "description": "BERT-base-cased. Wiki40b trainset first paragraph, freq=10~100, length=6~128."
    },
    "wiki40b-all": {
        "path":os.path.join(DIR_TRAIN_UNSUPERVISED, "bert-large-cased_wiki40b-train-all-paragraph_freq=10-100_len=6-128.hdf5"),
        "padding": False,
        "max_sequence_length": None,
        "description": "BERT-large-cased. Wiki40b trainset, freq=10~100, length=6~128."
    },
    "wiki40b-all-ext": {
        "path":os.path.join(DIR_TRAIN_UNSUPERVISED, "bert-large-cased_wiki40b-train-all-paragraph_freq=10-200_len=6-128.hdf5"),
        "padding": False,
        "max_sequence_length": None,
        "description": "BERT-large-cased. Wiki40b trainset, freq=10~200, length=6~128."
    },
    "SemCor": {
        "path":os.path.join(DIR_TRAIN, "NOT_YET_AVAILABLE.hdf5"),
        "padding": False,
        "max_sequence_length": None,
        "description": "BERT-large-cased. SemCor corpus."
    }
}