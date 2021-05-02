#!/usr/bin/env python
# -*- coding:utf-8 -*-

from typing import Optional, Dict, List, Union, Iterator
import os, sys, io, copy, glob2
from stanza.server import CoreNLPClient, StartServer
from stanza.protobuf.CoreNLP_pb2 import Document, Sentence, Token, Span
import gevent
import itertools
import multiprocessing

from ._mwe_parser_constant import _map_token_attrs_to_fields

def _extract_raw_sentence(doc: Document, sentence: Sentence):
    txt = doc.text
    idx_begin, idx_end = sentence.characterOffsetBegin, sentence.characterOffsetEnd
    return txt[idx_begin:idx_end]

def _extract_token_info(token: Token):
    dict_token_info = {field_name:getattr(token, attr_name, None) for attr_name, field_name in _map_token_attrs_to_fields.items()}

    # remove null fields
    for mwe_field in ("mwe_pos","mwe_lemma","mwe_span"):
        value = dict_token_info.get(mwe_field, "")
        if isinstance(value, str) and len(value) == 0:
            del dict_token_info[mwe_field]
        elif isinstance(value, Span) and value.ByteSize() == 0:
            del dict_token_info[mwe_field]
        elif value is None:
            del dict_token_info[mwe_field]

    # update span
    span = dict_token_info.get("mwe_span", None)
    if span is not None:
        dict_token_info["mwe_span"] = [span.begin, span.end]

    return dict_token_info

def _extract_sentence_tokens(sentence: Sentence):
    return list(map(_extract_token_info, sentence.token))

def parse_serialized_document(doc: Document):
    lst_ret = []
    for sentence in doc.sentence:
        dict_sentence = {
            "sentence": _extract_raw_sentence(doc, sentence),
            "token": _extract_sentence_tokens(sentence)
        }
        lst_ret.append(dict_sentence)
    return lst_ret


class CoreNLPTaggerWithMultiWordExpression(object):

    _default_corenlp_properties = {
        "annotators":"tokenize,ssplit,pos,lemma,jmwe",
        "customAnnotatorClass.jmwe":"edu.stanford.nlp.pipeline.JMWEAnnotator",
        "customAnnotatorClass.jmwe.verbose":"true",
        "customAnnotatorClass.jmwe.underscoreReplacement":"-",
        "customAnnotatorClass.jmwe.indexData":"./mweindex_wordnet3.0_semcor1.6.data",
        "customAnnotatorClass.jmwe.detector":"Consecutive"
    }

    def __init__(self, dir_corenlp: str, annotators: str = "tokenize,ssplit,pos,lemma,jmwe",
                 jmwe_detector_type: str = "Consecutive", output_format: str = "serialized",
                 kwargs_properties: Optional[Dict[str,str]] = None, kwargs_corenlp_client: Optional[Dict[str,str]] = None):
        _props = copy.deepcopy(self._default_corenlp_properties)
        _props["annotators"] = annotators
        _props["customAnnotatorClass.jmwe.detector"] = jmwe_detector_type
        if isinstance(kwargs_properties, dict):
            _props.update(kwargs_properties)

        # instanciate corenlp client
        _args = {
            "properties": _props,
            "output_format": output_format,
            "classpath": ":".join(glob2.glob(os.path.join(dir_corenlp, "*.jar"))),
            "start_server": StartServer.TRY_START
        }
        if isinstance(kwargs_corenlp_client, dict):
            _args.update(kwargs_corenlp_client)

        self._corenlp = CoreNLPClient(**_args)

        self._corenlp_properties = _props
        self._corenlp_client_args = _args

    def annotate_with_custom_format(self, document: str) -> Iterator[Dict[str, Union[str, int, None, List[int]]]]:
        # annotate with serialized format
        obj_doc = self._corenlp.annotate(document, output_format="serialized")
        return parse_serialized_document(obj_doc)

    def annotate_with_custom_format_parallel(self, lst_documents: List[str])-> Iterator[Dict[str, Union[str, int, None, List[int]]]]:
        threads = [gevent.spawn(self.annotate_with_custom_format, doc) for doc in lst_documents]
        lst_results = gevent.joinall(threads)
        iter_docs = (result.value for result in lst_results)
        all_sentences = itertools.chain(*iter_docs)
        return all_sentences

    # def annotate_with_custom_format_parallel(self, lst_documents: List[str])-> Iterator[Dict[str, Union[str, int, None, List[int]]]]:
    #     threads = [gevent.spawn(self.annotate_with_custom_format, doc) for doc in lst_documents]
    #     lst_results = gevent.joinall(threads)
    #     iter_docs = (result.value for result in lst_results)
    #     all_sentences = itertools.chain(*iter_docs)
    #     return all_sentences

    def annotate(self, document: str):
        return self._corenlp.annotate(document)


# exec stand-alone mode
if __name__ == "__main__":
    pass