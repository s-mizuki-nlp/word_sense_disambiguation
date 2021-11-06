#!/usr/bin/env python
# -*- coding:utf-8 -*-

from nltk.corpus import wordnet as wn

from typing import Dict, Any, Iterable, List, Tuple, Union
numeric = Union[int, float]
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

    def get_candidate_lemmas_from_wordnet(self, str_lemma: str, pos: str) -> List[wn.lemma]:
        lst_lemmas = wn.lemmas(str_lemma, pos=pos)
        assert len(lst_lemmas) > 0, f"unknown lemma: {str_lemma}|{pos}"
        return lst_lemmas

    def _print_verbose(self, lst_tup_lemma_and_score: List[Tuple[wn.lemma, Union[numeric, Tuple[numeric]]]]):
        print(f"metric: WordNet sense frequency")
        for lemma, score in lst_tup_lemma_and_score:
            print(f"{lemma.key()}: {score:3d}")

    def return_top_k_lemma_keys(self, lst_lemmas: List[wn.lemma], lst_scores: Union[List[numeric], Tuple[List[numeric]]],
                                multiple_output: bool) -> List[str]:
        if isinstance(lst_scores, tuple):
            lst_scores = list(zip(*lst_scores))
        lst_tup_lemma_and_scores = list(zip(lst_lemmas, lst_scores))
        lst_tup_lemma_and_scores = sorted(lst_tup_lemma_and_scores, key=lambda tup: tup[1], reverse=True)

        if self.verbose:
            self._print_verbose(lst_tup_lemma_and_scores)

        if multiple_output:
            lst_keys = []; prev_scores = None
            for lemma, scores in lst_tup_lemma_and_scores:
                if (prev_scores is not None) and (scores < prev_scores):
                    break
                lst_keys.append(lemma.key())
                prev_scores = scores
            return lst_keys
        else:
            lemma, scores = lst_tup_lemma_and_scores[0]
            return [lemma.key()]

    def score_by_sense_frequency(self, lst_lemmas: List[wn.lemma], reorder_by_lemma_count: bool = False) -> List[float]:
        if reorder_by_lemma_count:
            return [lemma.count() for lemma in lst_lemmas]
        else:
            # candidate order is used as it is.
            return list(range(0, -len(lst_lemmas), -1))

    def predict(self, input: Dict[str, Any], reorder_by_lemma_counts: bool = False, output_tie_lemma: bool = False) -> Iterable[str]:
        lst_lemmas = self.get_candidate_lemmas_from_wordnet(input["lemma"], input["pos"])
        lst_scores = self.score_by_sense_frequency(lst_lemmas, reorder_by_lemma_count=reorder_by_lemma_counts)
        lst_predicted = self.return_top_k_lemma_keys(lst_lemmas, lst_scores, multiple_output=output_tie_lemma)

        return lst_predicted
