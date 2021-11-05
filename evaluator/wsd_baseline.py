#!/usr/bin/env python
# -*- coding:utf-8 -*-

from nltk.corpus import wordnet as wn

from typing import Dict, Any, Iterable
from ._supervised_base import WSDTaskEvaluatorBase


# practical implementation for supervised task evaluation #

class ToyWSDTaskEvaluator(WSDTaskEvaluatorBase):

    def predict(self, input: Dict[str, Any], **kwargs):
        return ["art%1:09:00::"]


class MostFrequentSenseWSDTaskEvaluator(WSDTaskEvaluatorBase):

    def assertion(self):
        return True

    def get_lemma_count_from_wordnet(self, lemma_key: str, synset: str, **kwargs):
        ret = None
        try:
            ret = wn.lemma_from_key(lemma_key).count()
        except Exception as e:
            print(e)
            lst_lemmas = wn.synset(synset).lemmas()
            for lemma in lst_lemmas:
                if lemma.key() == lemma_key:
                    ret = lemma.count()
                    break

        if ret is None:
            raise ValueError(f"cannot find lemma: {synset}|{lemma_key}")
        return ret

    def predict_most_frequent_sense(self, str_lemma: str, pos: str, multiple_output: bool = False, reorder_by_lemma_counts: bool = False):
        lst_lemmas = wn.lemmas(str_lemma, pos=pos)
        assert len(lst_lemmas) > 0, f"unknown lemma: {str_lemma}|{pos}"

        if reorder_by_lemma_counts:
            lst_lemmas = sorted(lst_lemmas, key=lambda lemma: lemma.count(), reverse=True)

        if multiple_output:
            lst_keys = []; prev_freq = 0
            for lemma in lst_lemmas:
                if lemma.count() < prev_freq:
                    break
                lst_keys.append(lemma.key())
                prev_freq = lemma.count()
            return lst_keys
        else:
            return [lst_lemmas[0].key()]

    def predict(self, input: Dict[str, Any], reorder_by_lemma_counts: bool = False, output_tie_lemma: bool = False) -> Iterable[str]:
        return self.predict_most_frequent_sense(input["lemma"], input["pos"],
                                                reorder_by_lemma_counts=reorder_by_lemma_counts,
                                                multiple_output=output_tie_lemma)
