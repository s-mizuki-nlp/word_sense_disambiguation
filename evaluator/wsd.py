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

    def original_predict_most_frequent_sense(self, str_lemma: str, pos: str, multiple_output: bool = False):
        lst_lemmas = wn.lemmas(str_lemma, pos=pos)
        assert len(lst_lemmas) > 0, f"unknown lemma: {str_lemma}|{pos}"

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

    def bugfixed_predict_most_frequent_sense(self, input: Dict[str, Any], output_tie_lemma: bool = True) -> Iterable[str]:
        """
        wn.lemmas() may not return the lemmas ordered by frequency.
        this method look up lemma.count() and re-ranks.

        @param input:
        @param output_tie_lemma:
        @return:
        """
        lst_candidates = input["candidate_senses"]
        lst_lemma_freq = [self.get_lemma_count_from_wordnet(**sense) for sense in lst_candidates]
        lst_candidates_sorted = list(zip(lst_lemma_freq, lst_candidates))
        lst_candidates_sorted.sort(key=lambda tup: tup[0], reverse=True)

        if output_tie_lemma:
            lst_lemma_keys = []; prev_freq = 0
            for freq, sense in lst_candidates_sorted:
                if freq < prev_freq:
                    break
                lst_lemma_keys.append(sense["lemma_key"])
                prev_freq = freq
        else:
            _, sense = lst_candidates_sorted[0]
            lst_lemma_keys = [sense["lemma_key"]]

        return lst_lemma_keys

    def predict(self, input: Dict[str, Any], use_original_method: bool = True, output_tie_lemma: bool = False) -> Iterable[str]:
        if use_original_method:
            return self.original_predict_most_frequent_sense(input["lemma"], input["pos"], multiple_output=output_tie_lemma)
        else:
            return self.bugfixed_predict_most_frequent_sense(input, output_tie_lemma=output_tie_lemma)



