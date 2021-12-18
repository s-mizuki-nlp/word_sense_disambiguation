#!/usr/bin/env python
# -*- coding:utf-8 -*-

# from dataset.evaluation import EntityLevelWSDEvaluationDataset
from .wsd_task import CreateWSDTaskDataset, cfg_task_dataset
from . import sense_annotated_corpus, monosemous_corpus, lexical_knowledge_datasets

### pre-defined datasets ###
# wsd_eval_wo_embeddings = EntityLevelWSDEvaluationDataset(**sense_annotated_corpus.cfg_evaluation["WSDEval-ALL"])

wsd_eval_bert_large_cased = CreateWSDTaskDataset(
    cfg_bert_embeddings=sense_annotated_corpus.cfg_evaluation["WSDEval-ALL-bert-large-cased"],
    cfg_lemmas=None,
    cfg_synsets=None,
    **cfg_task_dataset["WSDEval"]
)

# synset code learningの評価用．WSDEval-noun-verbを使う．
wsd_validate_bert_large_cased = CreateWSDTaskDataset(
    cfg_bert_embeddings=sense_annotated_corpus.cfg_evaluation["WSDEval-noun-verb-bert-large-cased"],
    cfg_lemmas=lexical_knowledge_datasets.cfg_lemma_datasets["WordNet-noun-verb-incl-instance"],
    cfg_synsets=lexical_knowledge_datasets.cfg_synset_datasets["WordNet-noun-verb-incl-instance"],
    **cfg_task_dataset["WSDValidation"]
)

try:
    wsd_train_wikitext103_subset = CreateWSDTaskDataset(
        cfg_bert_embeddings=monosemous_corpus.cfg_training["wikitext103-subset"],
        cfg_lemmas=lexical_knowledge_datasets.cfg_lemma_datasets["WordNet-noun-verb-incl-instance"],
        cfg_synsets=lexical_knowledge_datasets.cfg_synset_datasets["WordNet-noun-verb-incl-instance"],
        **cfg_task_dataset["TrainOnMonosemousCorpus"]
    )

    wsd_train_wiki40b_all = CreateWSDTaskDataset(
        cfg_bert_embeddings=monosemous_corpus.cfg_training["wiki40b-all"],
        cfg_lemmas=lexical_knowledge_datasets.cfg_lemma_datasets["WordNet-noun-verb-incl-instance"],
        cfg_synsets=lexical_knowledge_datasets.cfg_synset_datasets["WordNet-noun-verb-incl-instance"],
        **cfg_task_dataset["TrainOnMonosemousCorpus"]
    )

    wsd_train_wiki40b_all_ext = CreateWSDTaskDataset(
        cfg_bert_embeddings=monosemous_corpus.cfg_training["wiki40b-all-ext"],
        cfg_lemmas=lexical_knowledge_datasets.cfg_lemma_datasets["WordNet-noun-verb-incl-instance"],
        cfg_synsets=lexical_knowledge_datasets.cfg_synset_datasets["WordNet-noun-verb-incl-instance"],
        **cfg_task_dataset["TrainOnMonosemousCorpus"]
    )

    wsd_train_wiki40b_all_wide_vocab = CreateWSDTaskDataset(
        cfg_bert_embeddings=monosemous_corpus.cfg_training["wiki40b-all-wide-vocab"],
        cfg_lemmas=lexical_knowledge_datasets.cfg_lemma_datasets["WordNet-noun-verb-incl-instance"],
        cfg_synsets=lexical_knowledge_datasets.cfg_synset_datasets["WordNet-noun-verb-incl-instance"],
        **cfg_task_dataset["TrainOnMonosemousCorpus"]
    )

    wsd_train_wiki40b_all_narrow_vocab = CreateWSDTaskDataset(
        cfg_bert_embeddings=monosemous_corpus.cfg_training["wiki40b-all-narrow-vocab"],
        cfg_lemmas=lexical_knowledge_datasets.cfg_lemma_datasets["WordNet-noun-verb-incl-instance"],
        cfg_synsets=lexical_knowledge_datasets.cfg_synset_datasets["WordNet-noun-verb-incl-instance"],
        **cfg_task_dataset["TrainOnMonosemousCorpus"]
    )

except Exception as e:
    print(e)
