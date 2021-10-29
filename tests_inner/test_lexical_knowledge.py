#!/usr/bin/env python
# -*- coding:utf-8 -*-

import io
import unittest

from nltk.corpus import wordnet as wn

from dataset.lexical_knowledge import SynsetDataset, LemmaDataset
from config_files.lexical_knowledge_datasets import cfg_synset_datasets, cfg_lemma_datasets


class LemmaDatasetTestCases(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls._dataset = LemmaDataset(**cfg_lemma_datasets["WordNet-noun-verb-incl-instance"])

    def test_lookup_synset_id_from_lemma_key(self):
        for record in self._dataset:
            for lemma_key in record["lemma_keys"].keys():
                try:
                    wn_lemma = wn.lemma_from_key(lemma_key)
                except Exception as e:
                    print(f"skip: {lemma_key}")
                    continue

                with self.subTest(lemma_key=lemma_key):
                    expected = wn_lemma.synset().name()
                    actual = self._dataset.get_synset_id_from_lemma_key(lemma_key)
                    self.assertEqual(expected, actual)

    def test_lookup_lexname_from_lemma_key(self):
        for record in self._dataset:
            for lemma_key in record["lemma_keys"].keys():
                try:
                    wn_lemma = wn.lemma_from_key(lemma_key)
                except Exception as e:
                    print(f"skip: {lemma_key}")
                    continue

                with self.subTest(lemma_key=lemma_key):
                    expected = wn_lemma.synset().lexname()
                    actual = self._dataset.get_lexname_from_lemma_key(lemma_key)
                    self.assertEqual(expected, actual)


class SynsetDatasetTestCases(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls._dataset = SynsetDataset(**cfg_synset_datasets["WordNet-noun-verb-incl-instance"])

    def test_lemma_keys_lookup(self):
        lst_synset_ids = ["play.n.01", "earth.n.01", "play.v.01"]

        def _lookup_lemmas(synset_id: str):
            return [lemma.key() for lemma in wn.synset(synset_id).lemmas()]

        for synset_id in lst_synset_ids:
            with self.subTest(synset_id=synset_id):
                expected = set(_lookup_lemmas(synset_id))
                actual = set(self._dataset.lookup_lemma_keys(synset_id))
                self.assertSetEqual(expected, actual)

    def test_code_lookup(self):
        lst_synset_ids = ["play.n.01", "earth.n.01", "play.v.01"]

        for synset_id in lst_synset_ids:
            with self.subTest(synset_id=synset_id):
                code = self._dataset[synset_id]["code"]
                expected = synset_id
                actual = self._dataset[code]["id"]
                self.assertEqual(expected, actual)
