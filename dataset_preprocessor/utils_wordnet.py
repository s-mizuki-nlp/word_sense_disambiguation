#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import numpy as np
import nltk
from nltk.corpus import wordnet as wn

def is_isolated_synset(synset):
    return len(synset.hypernyms()) + len(synset.hyponyms()) == 0

def synset_to_name(synset):
    return synset.name()

def synset_to_lexeme(synset):
    pos = synset.pos()
    return synset.name().split(f".{pos}")[0]

def synset_to_lemmas(synset, include_phrase):
    lst_ret = synset.lemma_names()
    if not include_phrase:
        lst_ret = [lemma for lemma in lst_ret if lemma.find("_") == -1]
    return lst_ret

def get_hypernym_synsets_and_distance_via_shortest_path(synset):
    dict_ret = {}
    for hypernym, distance in synset.hypernym_distances():
        if hypernym == synset:
            continue
        if hypernym.pos() != synset.pos():
            print(hypernym, synset)
            continue
        current_value = dict_ret.get(hypernym, float("inf"))
        if distance < current_value:
            dict_ret[hypernym] = distance

    return dict_ret

def _list_up_depth_to_root(synset, root_synset):
    ret = []
    for s, d in synset.hypernym_distances():
        if s == root_synset:
            ret.append(d)
    return ret

def get_synset_depth(synset, root_synset={"n":wn.synset("entity.n.01"), "v":None}):
    ret_empty = (None, None, None)
    if isinstance(root_synset, dict):
        root_synset = root_synset.get(synset.pos(), None)
    if root_synset is None:
        return ret_empty

    # returns max, min, avg depth
    lst_depth = _list_up_depth_to_root(synset, root_synset)
    if len(lst_depth) == 0:
        return ret_empty
    else:
        return (float(np.max(lst_depth)), float(np.min(lst_depth)), float(np.mean(lst_depth)))


_MAP_UNIVERSAL_TO_WORDNET = {
    "NOUN": wn.NOUN,
    "VERB": wn.VERB,
    "ADJ": wn.ADJ,
    "ADV": wn.ADV
}
_MAP_PTB_TO_UNIVERSAL = nltk.tagset_mapping("en-ptb", "universal")

def universal_tagset_to_wordnet_tagset(univ_tag, na_value = "o"):
    """
    Universal PoS tagset を WordNetの PoS tagset に変換する．
    @param univ_tag: Universal PoS tag
    @param na_value: 変換先のWordNet PoS tagがない場合の出力
    @return:　WordNet PoS tagset. {wn.NOUN, wn.VERB, wn.ADJ, wn.ADV, na_value} のいずれか
    """
    wn_tag = _MAP_UNIVERSAL_TO_WORDNET.get(univ_tag, na_value)
    return wn_tag

def ptb_tagset_to_wordnet_tagset(ptb_tag, na_value = "o"):
    """
    Penn TreeBank PoS tagset を WordNetの PoS tagset に変換する．
    @param ptb_tag: Penn TreeBank PoS tag
    @param na_value: 変換先のWordNet PoS tagがない場合の出力
    @return:　WordNet PoS tagset. {wn.NOUN, wn.VERB, wn.ADJ, wn.ADV, na_value} のいずれか
    """
    univ_tag = _MAP_PTB_TO_UNIVERSAL[ptb_tag]
    return universal_tagset_to_wordnet_tagset(univ_tag, na_value)

def lemma_to_surface_form(lemma: wn.lemma):
    return lemma.name().replace('_', ' ')

