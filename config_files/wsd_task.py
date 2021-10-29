#!/usr/bin/env python
# -*- coding:utf-8 -*-

from typing import Dict, Any, Optional

from torch.utils.data import DataLoader

from dataset import WSDTaskDataset, WSDTaskDatasetCollateFunction
from dataset.lexical_knowledge import LemmaDataset, SynsetDataset
from dataset.contextualized_embeddings import BERTEmbeddingsDataset


def CreateWSDTaskDataset(cfg_bert_embeddings: Dict[str, Any],
                         cfg_lemmas: Dict[str, Any] = None,
                         cfg_synsets: Optional[Dict[str, Any]] = None,
                         **kwargs):
    dataset_bert_embeddings = BERTEmbeddingsDataset(**cfg_bert_embeddings)
    dataset_lemmas = LemmaDataset(**cfg_lemmas) if cfg_lemmas is not None else None
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

