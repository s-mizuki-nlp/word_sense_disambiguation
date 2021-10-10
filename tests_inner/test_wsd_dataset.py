#!/usr/bin/env python
# -*- coding:utf-8 -*-

import io
import unittest

from nltk.corpus import wordnet as wn
from bs4 import BeautifulSoup

from dataset.evaluation import WSDEvaluationDataset
from config_files.wsd_datasets import word_sense_disambiguation


class WSDEvaluationDatasetTestCases(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls._dataset = WSDEvaluationDataset(**word_sense_disambiguation["ALL"])

    def test_xml_parse(self):

        str_xml = """
            <sentence id="semeval2013.d000.s002">
            <wf lemma="the" pos="DET">The</wf>
            <instance id="semeval2013.d000.s002.t000" lemma="text" pos="NOUN">text</instance>
            <wf lemma="," pos=".">,</wf>
            <wf lemma="which" pos="DET">which</wf>
            <wf lemma="could" pos="VERB">could</wf>
            <wf lemma="provide" pos="VERB">provide</wf>
            <wf lemma="the" pos="DET">the</wf>
            <instance id="semeval2013.d000.s002.t001" lemma="basis" pos="NOUN">basis</instance>
            <wf lemma="for" pos="ADP">for</wf>
            <wf lemma="a" pos="DET">a</wf>
            <wf lemma="final" pos="ADJ">final</wf>
            <wf lemma="political" pos="ADJ">political</wf>
            <instance id="semeval2013.d000.s002.t002" lemma="deal" pos="NOUN">deal</instance>
            <wf lemma="to" pos="PRT">to</wf>
            <wf lemma="regulate" pos="VERB">regulate</wf>
            <instance id="semeval2013.d000.s002.t003" lemma="greenhouse_gas" pos="NOUN">greenhouse gases</instance>
            <wf lemma="," pos=".">,</wf>
            <wf lemma="highlight" pos="VERB">highlighted</wf>
            <wf lemma="the" pos="DET">the</wf>
            <wf lemma="remain" pos="VERB">remaining</wf>
            <instance id="semeval2013.d000.s002.t004" lemma="obstacle" pos="NOUN">obstacles</instance>
            <wf lemma="as" pos="ADV">as</wf>
            <wf lemma="much" pos="ADV">much</wf>
            <wf lemma="as" pos="ADP">as</wf>
            <wf lemma="it" pos="PRON">it</wf>
            <wf lemma="illuminate" pos="VERB">illuminated</wf>
            <wf lemma="a" pos="DET">a</wf>
            <instance id="semeval2013.d000.s002.t005" lemma="path" pos="NOUN">path</instance>
            <wf lemma="forward" pos="ADV">forward</wf>
            <wf lemma="." pos=".">.</wf>
            </sentence>
        """

        with io.StringIO(str_xml) as ifs:
            soup = BeautifulSoup(ifs, "lxml")
        sentence_node = soup.select_one("sentence")
        dict_parsed = self._dataset._parse_sentence_node(sentence_node)

        # entities
        expected = str_xml.count("<instance")
        actual = len(dict_parsed["entities"])
        self.assertEqual(expected, actual)

        # surfaces
        expected = str_xml.count("<instance") + str_xml.count("<wf")
        actual = len(dict_parsed["surfaces"])
        self.assertEqual(expected, actual)

    def test_sequence_length(self):
        for record in self._dataset:
            with self.subTest(sentence_id=record["sentence_id"]):
                expected = record["tokenized_sentence"].count(" ") + 1
                actual = len(record["words"])
                self.assertEqual(expected, actual)

    def test_sequence(self):
        for record in self._dataset:
            with self.subTest(sentence_id=record["sentence_id"]):
                expected = record["tokenized_sentence"]
                actual = " ".join(record["words"])
                self.assertEqual(expected, actual)

    def test_corpus_id(self):
        expected = set("senseval2,senseval3,semeval2007,semeval2013,semeval2015".split(","))
        actual = set(self._dataset.distinct_values(column="corpus_id"))
        self.assertSetEqual(expected, actual)

    def test_entity_counts(self):
        expected = len(self._dataset._ground_truth_labels)
        actual = 0
        for record in self._dataset:
            actual += len(record["entities"])
        self.assertEqual(expected, actual)

    def test_entity_pos(self):
        expected = {wn.NOUN, wn.VERB, wn.ADJ, wn.ADV}
        for record in self._dataset:
            with self.subTest(sentence_id=record["sentence_id"]):
                actual = set([entity["pos"] for entity in record["entities"]])
                self.assertTrue(actual.issubset(expected))

    def test_candidate_senses(self):
        for record in self._dataset:
            for entity in record["entities"]:
                with self.subTest(sentence_id=record["sentence_id"]):
                    ground_truths = set(entity["ground_truth_lemma_keys"])
                    candidates = set([c["lemma_key"] for c in entity["candidate_senses"]])
                    self.assertTrue(ground_truths.issubset(candidates))
                    self.assertGreater(len(candidates), 0)

    def test_sense_lookup(self):
        lst_lemmas = wn.lemmas("cat", pos="n")
        expected = set(lemma.key() for lemma in lst_lemmas)

        lst_candidates = self._dataset._lookup_candidate_senses_from_wordnet(lemma="cat", pos="n")
        actual = set(c["lemma_key"] for c in lst_candidates)

        self.assertSetEqual(expected, actual)


if __name__ == '__main__':
    unittest.main()
