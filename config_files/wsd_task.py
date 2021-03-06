#!/usr/bin/env python
# -*- coding:utf-8 -*-

from typing import Dict, Any, Optional, Union, List

from torch.utils.data import DataLoader, BufferedShuffleDataset, ChainDataset

from dataset import WSDTaskDataset, WSDTaskDatasetCollateFunction
from dataset.lexical_knowledge import LemmaDataset, SynsetDataset
from dataset.contextualized_embeddings import BERTEmbeddingsDataset

USE_ENTITY_EMBEDDING = 0.0
USE_SENTENCE_EMBEDDING = 1.0


def CreateWSDTaskDataset(cfg_bert_embeddings: Union[Dict[str, Any], List[Dict[str, Any]]],
                         cfg_lemmas: Dict[str, Any] = None,
                         cfg_synsets: Optional[Dict[str, Any]] = None,
                         **kwargs):
    if isinstance(cfg_bert_embeddings, dict):
        cfg_bert_embeddings = [cfg_bert_embeddings]

    lst_dataset_bert_embeddings = [BERTEmbeddingsDataset(**cfg_) for cfg_ in cfg_bert_embeddings]
    dataset_lemmas = LemmaDataset(**cfg_lemmas) if cfg_lemmas is not None else None
    dataset_synsets = SynsetDataset(**cfg_synsets) if cfg_synsets is not None else None

    if len(lst_dataset_bert_embeddings) == 1:
        dataset_bert_embeddings = lst_dataset_bert_embeddings[0]
    else:
        dataset_bert_embeddings = ChainDataset(lst_dataset_bert_embeddings)

    dataset = WSDTaskDataset(
        bert_embeddings_dataset=dataset_bert_embeddings,
        lexical_knowledge_lemma_dataset=dataset_lemmas,
        lexical_knowledge_synset_dataset=dataset_synsets,
        **kwargs
    )
    return dataset


def WSDTaskDataLoader(dataset: Union[WSDTaskDataset, BufferedShuffleDataset],
                      batch_size: int,
                      cfg_collate_function: Dict[str, Any] = {},
                      **kwargs):
    if "is_trainset" not in cfg_collate_function:
        if isinstance(dataset, BufferedShuffleDataset):
            is_trainset = getattr(dataset.dataset, "is_trainset", False)
        else:
            is_trainset = dataset.is_trainset
        cfg_collate_function["is_trainset"] = is_trainset
    collate_fn = WSDTaskDatasetCollateFunction(**cfg_collate_function)
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, collate_fn=collate_fn, **kwargs)

    return data_loader


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
    },
    "TrainOnWordNetGlossCorpus": {
        "is_trainset": True,
        "return_level":"entity",
        "record_entity_field_name":"entities",
        "record_entity_span_field_name":"subword_spans",
        "ground_truth_lemma_keys_field_name":"ground_truth_lemma_keys",
        "copy_field_names_from_record_to_entity":None,
        "return_entity_subwords_avg_vector":True,
        "raise_error_on_unknown_lemma":True,
        "weighted_average_entity_embeddings_and_sentence_embedding":USE_ENTITY_EMBEDDING,
        "normalize_embeddings":False
    }
}