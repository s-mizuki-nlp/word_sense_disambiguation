#!/usr/bin/env python
# -*- coding:utf-8 -*-

__straight_through_mapping_fields = "word,pos,lemma"

__customized_mapping_fields = {
    "trueCase":"mwe_pos",
    "trueCaseText":"mwe_lemma",
    "span":"mwe_span"
}

_map_token_attrs_to_fields = {v:v for v in __straight_through_mapping_fields.split(",")}
_map_token_attrs_to_fields.update(__customized_mapping_fields)
