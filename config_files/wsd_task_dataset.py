#!/usr/bin/env python
# -*- coding:utf-8 -*-

from typing import Dict, Any, Optional

from . import sense_annotated_corpus, monosemous_corpus, lexical_knowledge_datasets

from torch.utils.data import DataLoader

from dataset import WSDTaskDataset, WSDTaskDatasetCollateFunction
from dataset.lexical_knowledge import LemmaDataset, SynsetDataset
from dataset.evaluation import EntityLevelWSDEvaluationDataset
from dataset.contextualized_embeddings import BERTEmbeddingsDataset


cfg_task_dataset = {
    "WSDEval": {
        "is_trainset": False,
        "return_level":"entity",
        "record_entity_field_name":"entities",
        "record_entity_span_field_name":"subword_spans",
        "copy_field_names_from_record_to_entity":["corpus_id","document_id","sentence_id","words"],
        "return_entity_subwords_avg_vector":True,
        "raise_error_on_unknown_lemma":False
    },
    "WSDValidation": {
        "is_trainset": True,
        "return_level":"entity",
        "record_entity_field_name":"entities",
        "record_entity_span_field_name":"subword_spans",
        "ground_truth_lemma_keys_field_name":"ground_truth_lemma_keys",
        "copy_field_names_from_record_to_entity":["corpus_id","document_id","sentence_id","words"],
        "return_entity_subwords_avg_vector":True,
        "raise_error_on_unknown_lemma":True
    },
    "TrainOnMonosemousCorpus": {
        "is_trainset": True,
        "return_level":"entity",
        "record_entity_field_name":"monosemous_entities",
        "record_entity_span_field_name":"subword_spans",
        "copy_field_names_from_record_to_entity":None,
        "return_entity_subwords_avg_vector":True,
        "raise_error_on_unknown_lemma":True
    }
}


def CreateWSDEvaluationTaskDataset(cfg_bert_embeddings: Dict[str, Any],
                                   **kwargs):
    dataset_bert_embeddings = BERTEmbeddingsDataset(**cfg_bert_embeddings)

    dataset = WSDTaskDataset(
        bert_embeddings_dataset=dataset_bert_embeddings,
        lexical_knowledge_lemma_dataset=None,
        lexical_knowledge_synset_dataset=None,
        **kwargs
    )
    return dataset

def CreateWSDTrainingTaskDataset(cfg_bert_embeddings: Dict[str, Any],
                                 cfg_lemmas: Dict[str, Any],
                                 cfg_synsets: Optional[Dict[str, Any]] = None,
                                 **kwargs):
    dataset_bert_embeddings = BERTEmbeddingsDataset(**cfg_bert_embeddings)
    dataset_lemmas = LemmaDataset(**cfg_lemmas)
    dataset_synsets = SynsetDataset(**cfg_synsets) if cfg_synsets is not None else None

    dataset = WSDTaskDataset(
        bert_embeddings_dataset=dataset_bert_embeddings,
        lexical_knowledge_lemma_dataset=dataset_lemmas,
        lexical_knowledge_synset_dataset=dataset_synsets,
        **kwargs
    )
    return dataset


def WSDTaskDataLoader(dataset: WSDTaskDataset,
                      batch_size: int,
                      cfg_collate_function: Dict[str, Any] = {},
                      **kwargs):
    if "is_trainset" not in cfg_collate_function:
        cfg_collate_function["is_trainset"] = dataset.is_trainset
    collate_fn = WSDTaskDatasetCollateFunction(**cfg_collate_function)
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, collate_fn=collate_fn, **kwargs)

    return data_loader

### pre-defined datasets ###
wsd_eval_wo_embeddings = EntityLevelWSDEvaluationDataset(**sense_annotated_corpus.cfg_evaluation["WSDEval-ALL"])

wsd_eval_bert_large_cased = CreateWSDEvaluationTaskDataset(
    cfg_bert_embeddings=sense_annotated_corpus.cfg_evaluation["WSDEval-ALL-bert-large-cased"],
    **cfg_task_dataset["WSDEval"]
)

wsd_train_wikitext103_subset = CreateWSDTrainingTaskDataset(
    cfg_bert_embeddings=monosemous_corpus.cfg_training["wikitext103-subset"],
    cfg_lemmas=lexical_knowledge_datasets.cfg_lemma_datasets["WordNet-noun-verb-incl-instance"],
    cfg_synsets=lexical_knowledge_datasets.cfg_synset_datasets["WordNet-noun-verb-incl-instance"],
    **cfg_task_dataset["TrainOnMonosemousCorpus"]
)

# synset code learningの評価用．WSDEval-noun-verbを使う．
wsd_validate_bert_large_cased = CreateWSDTrainingTaskDataset(
    cfg_bert_embeddings=sense_annotated_corpus.cfg_evaluation["WSDEval-noun-verb-bert-large-cased"],
    cfg_lemmas=lexical_knowledge_datasets.cfg_lemma_datasets["WordNet-noun-verb-incl-instance"],
    cfg_synsets=lexical_knowledge_datasets.cfg_synset_datasets["WordNet-noun-verb-incl-instance"],
    **cfg_task_dataset["WSDValidation"]
)