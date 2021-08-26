#!/usr/bin/env python
# -*- coding:utf-8 -*-

from typing import Set, Optional
from .contextualized_embeddings import BERTEmbeddingsDataset
from .lexical_knowledge import LemmaDataset

from torch.utils.data import IterableDataset
from .encoder import extract_entity_subword_embeddings, calc_entity_embeddings_from_subword_embeddings


class TrainingDataset(IterableDataset):

    def __init__(self, bert_embeddings_dataset: BERTEmbeddingsDataset,
                 lexical_knowledge_lemma_dataset: LemmaDataset,
                 expand_entities: bool = True,
                 return_entity_vector: bool = False,
                 additional_corpus_fields: Optional[Set[str]] = None):

        self._bert_embeddings = bert_embeddings_dataset
        self._lexical_knowledge_lemma = lexical_knowledge_lemma_dataset
        self._expand_entities = expand_entities
        self._return_entity_vector = return_entity_vector
        self._additional_fields = set() if additional_corpus_fields is None else additional_corpus_fields

    def _expand_entities(self):
        pass


    def _entity_loader(self):
        it_sentences = self._sentence_loader()
        for obj_sentence in it_sentences:
            for monosemous_entity in obj_sentence["monosemous_entities"]:
                context_embeddings = obj_sentence["embedding"]
                context_sequence_length = obj_sentence["sequence_length"]

    def _sentence_loader(self):
        for obj_bert_embeddings in self._bert_embeddings:
            lst_records = obj_bert_embeddings["records"]
            obj_entity_embeddings = extract_entity_subword_embeddings(
                                        context_embeddings=obj_bert_embeddings["embeddings"],
                                        lst_lst_entity_subword_spans=)
            yield obj_bert_embeddings

    def __iter__(self):
        if self._expand_entities:
            it_records = self._entity_loader()
        else:
            it_records = self._sentence_loader()
        for record in it_records:
            yield record

    @property
    def lexical_knowledge_lemma_dataset(self):
        return self._lexical_knowledge_lemma

    @property
    def bert_embeddings_dataset(self):
        return self._bert_embeddings