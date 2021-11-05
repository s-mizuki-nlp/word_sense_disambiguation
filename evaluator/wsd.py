#!/usr/bin/env python
# -*- coding:utf-8 -*-

from typing import Dict, Any, Iterable, Optional, Set, Union, Tuple, List
import torch
from nltk.corpus import wordnet as wn

from model.core import HierarchicalCodeEncoder
from model.loss_supervised import HyponymyScoreLoss

from .wsd_baseline import MostFrequentSenseWSDTaskEvaluator, WSDTaskEvaluatorBase
from dataset import WSDTaskDataset
from dataset.lexical_knowledge import SynsetDataset
from dataset.utils import tensor_to_numpy


class SenseCodeWSDTaskEvaluator(MostFrequentSenseWSDTaskEvaluator):

    __AVAILABLE_METRICS = ("common_prefix_length","common_prefix_length_ratio", "inclusion_probability", "synonym_probability")

    def __init__(self,
                 model: HierarchicalCodeEncoder,
                 evaluation_dataset: WSDTaskDataset,
                 inference_metric: str,
                 lexical_knowledge_synset_dataset: Optional[SynsetDataset] = None,
                 target_pos: Tuple[str] = ("n","v"),
                 ground_truth_lemma_keys_field_name: str = "ground_truth_lemma_keys",
                 breakdown_attributes: Optional[Iterable[Set[str]]] = None,
                 verbose: bool = False,
                 **kwargs_dataloader):
        super().__init__(evaluation_dataset, ground_truth_lemma_keys_field_name, breakdown_attributes, verbose, **kwargs_dataloader)

        self._aux_hyponymy_score = HyponymyScoreLoss(log_scale=False)
        self._model = model
        self._lexical_knowledge_synset = evaluation_dataset.synset_dataset if lexical_knowledge_synset_dataset is None else lexical_knowledge_synset_dataset
        self._inference_metric = inference_metric
        self._target_pos = target_pos

        assert inference_metric in self.__AVAILABLE_METRICS, f"`inference_metric` must be: {self.__AVAILABLE_METRICS}"
        assert self._lexical_knowledge_synset is not None, f"You must specify `lexical_knowledge_synset_dataset` argument."

    def calc_common_prefix_lengths(self, predicted_code_probs: torch.Tensor, candidate_codes: torch.Tensor, return_ratio: bool) -> torch.Tensor:
        # predicted_code_probs: (n_candidate, n_digits, n_ary)
        # candidate_codes: (n_candidate, n_digits, n_ary)

        # candidate code lengths
        t_code_length_candidates = (candidate_codes.argmax(dim=-1) != 0).sum(axis=-1).type(torch.float)
        # common prefix lengths
        t_soft_cpl = self._aux_hyponymy_score.calc_soft_lowest_common_ancestor_length(t_prob_c_x=candidate_codes, t_prob_c_y=predicted_code_probs)
        # ratio
        t_soft_cpl_ratio = t_soft_cpl / t_code_length_candidates

        if return_ratio:
            return t_soft_cpl_ratio
        else:
            return t_soft_cpl

    def calc_inclusion_probability(self, predicted_code_probs: torch.Tensor, candidate_codes: torch.Tensor, add_entailment_probs: bool) -> torch.Tensor:
        # predicted_code_probs: (n_candidate, n_digits, n_ary)
        # candidate_codes: (n_candidate, n_digits, n_ary)

        # entailment probability and synonym probability
        t_prob_entail = self._aux_hyponymy_score.calc_ancestor_probability(t_prob_c_x=candidate_codes, t_prob_c_y=predicted_code_probs)
        t_prob_synonym = self._aux_hyponymy_score.calc_synonym_probability(t_prob_c_x=candidate_codes, t_prob_c_y=predicted_code_probs)
        if add_entailment_probs:
            t_prob_inclusion = t_prob_synonym + t_prob_entail
        else:
            t_prob_inclusion = t_prob_synonym

        return t_prob_inclusion

    def score_by_inference_metric(self, candidate_codes, predicted_code_prob: torch.Tensor) -> List[float]:
        # candidate_codes: (n_candidates, n_digits)
        # predicted_code_prob: (1, n_digits, n_ary)

        # reshaping
        t_candidate_codes = self._aux_hyponymy_score._one_hot_encoding(t_codes=candidate_codes, n_ary=self._model.n_ary, label_smoothing_factor=0.0)

        n_candidates = t_candidate_codes.shape[0]
        # predicted_code_probs: (n_candidates, n_digits, n_ary)
        predicted_code_probs = torch.tile(predicted_code_prob, (n_candidates,1,1))

        # compute similarity score: bigger is better.
        if self._inference_metric == "common_prefix_length":
            t_scores = self.calc_common_prefix_lengths(predicted_code_probs=predicted_code_probs,
                                                         candidate_codes=t_candidate_codes, return_ratio=False)
        elif self._inference_metric == "common_prefix_length_ratio":
            t_scores = self.calc_common_prefix_lengths(predicted_code_probs=predicted_code_probs,
                                                         candidate_codes=t_candidate_codes, return_ratio=True)
        elif self._inference_metric == "inclusion_probability":
            t_scores = self.calc_inclusion_probability(predicted_code_probs=predicted_code_probs,
                                                         candidate_codes=t_candidate_codes, add_entailment_probs=True)
        elif self._inference_metric == "synonym_probability":
            t_scores = self.calc_inclusion_probability(predicted_code_probs=predicted_code_probs,
                                                         candidate_codes=t_candidate_codes, add_entailment_probs=False)

        return tensor_to_numpy(t_scores).tolist()

    def predict(self, input: Dict[str, Any],
                apply_greedy_decoding: bool = False,
                apply_one_hot_encoding: bool = False,
                mfs_reorder_by_lemma_counts: bool = False,
                output_tie_lemma: bool = False) -> Iterable[str]:
        """
        predict the most plausible sense using using

        @param input:
        @param apply_greedy_decoding:
        @param apply_one_hot_encoding: convert from continuous relaxed repr. to one-hot repr.
        @return:
        """
        lemma = input["lemma"]
        pos = input["pos"]

        # if unsupported part-of-speech tag, then fall back to most frequent sense method.
        if pos not in self._target_pos:
            return super().predict(input, reorder_by_lemma_counts=mfs_reorder_by_lemma_counts, output_tie_lemma=output_tie_lemma)

        # if supported part-of-speech tag, then rank the candidates using predicted code probabilities.
        # do inference
        _, t_code_prob = self._model._predict(**input, apply_argmax_on_inference=apply_greedy_decoding)
        if apply_one_hot_encoding:
            t_code_prob = t_code_prob.argmax(dim=-1)
            t_code_prob = self._aux_hyponymy_score._one_hot_encoding(t_codes=t_code_prob, n_ary=self._model.n_ary, label_smoothing_factor=0.0)
        # get candidates
        lst_candidate_lemmas = self.get_candidate_lemmas_from_wordnet(lemma, pos)
        lst_synset_ids = [lemma.synset().name() for lemma in lst_candidate_lemmas]
        lst_synset_codes = [self._lexical_knowledge_synset.get_synset_code(synset_id) for synset_id in lst_synset_ids]
        # t_candidate_codes: (n_candidates, n_digits, n_ary)
        t_candidate_codes = torch.LongTensor(lst_synset_codes, device=t_code_prob.device)
        # calc score for each candidate using specified inference metric.
        lst_scores = self.score_by_inference_metric(candidate_codes=t_candidate_codes, predicted_code_prob=t_code_prob)
        # return top-k lemma keys
        return self.return_top_k_lemma_keys(lst_candidate_lemmas, lst_scores, multiple_output=output_tie_lemma)

    def _print_verbose(self, lst_tup_lemma_and_score: List[Tuple[wn.lemma, float]]):
        print(f"metric: {self._inference_metric}")
        for lemma, score in lst_tup_lemma_and_score:
            synset_id = lemma.synset().name()
            sense_code = self._lexical_knowledge_synset.get_synset_code(synset_id)
            str_sense_code = "-".join(map(str, sense_code))
            print(f"{synset_id}-{lemma.key()}: {score:1.6f}, {str_sense_code}")
