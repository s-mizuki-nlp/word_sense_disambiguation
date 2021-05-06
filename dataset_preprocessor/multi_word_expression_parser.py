#!/usr/bin/env python
# -*- coding:utf-8 -*-

from typing import Optional, Dict, List, Union, Iterator
import os, sys, io, copy, glob2
from stanza.server import CoreNLPClient, StartServer
from stanza.protobuf.CoreNLP_pb2 import Document, Sentence, Token, Span
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


# def _init_annotate_parse(kwargs, doc: str, index: int):
#     """
#     並列処理用のヘルパー関数
#
#     @param kwargs: corenlp clientの引数．一部は強制的に上書きされる
#     @param doc: 処理するdocument．できるだけ大きくするとよい
#     @param index: プロセス番号(0,1,2,...,n_process)
#     @return: 処理したドキュメント
#     """
#     kwargs["start_server"] = StartServer.TRY_START
#     kwargs["endpoint"] = f"http://localhost:{9000+index}"
#     kwargs["output_format"] = "serialized"
#     kwargs["threads"] = 1
#     kwargs["be_quiet"] = False
#
#     # わざとclientを終了しない．すると2回目以降の呼び出しで既存clientを再利用してくれる
#     client = CoreNLPClient(**kwargs)
#     obj_doc = client.annotate(doc)
#     iter_sentences = parse_serialized_document(obj_doc)
#     return iter_sentences


def _annotate_parse(client: CoreNLPClient, doc: str):
    """
    並列処理用のヘルパー関数

    @param client: corenlp clientインスタンス
    @param doc: 処理するdocument．できるだけ大きくするとよい
    @param index: プロセス番号(0,1,2,...,n_process)
    @return: 処理したドキュメント
    """

    obj_doc = client.annotate(doc)
    iter_sentences = parse_serialized_document(obj_doc)
    return iter_sentences


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
                 threads: int = 1,
                 kwargs_properties: Optional[Dict[str,str]] = None, kwargs_corenlp_client: Optional[Dict[str,str]] = None):
        """
        Stanford Core NLP Client Wrapper class.

        @param dir_corenlp:
        @param annotators:
        @param jmwe_detector_type:
        @param output_format:
        @param kwargs_properties:
        @param kwargs_corenlp_client:
        """
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


        self._corenlp = {}
        if threads == 1:
            self._corenlp[0] = CoreNLPClient(**_args)
        else:
            self._pool = multiprocessing.Pool(threads)
            for index in range(threads):
                _args_i = copy.deepcopy(_args)
                _args_i["start_server"] = StartServer.TRY_START
                _args_i["endpoint"] = f"http://localhost:{9000+index}"
                _args_i["output_format"] = "serialized"
                _args_i["threads"] = 1
                self._corenlp[index] = CoreNLPClient(**_args_i)

        self._corenlp_properties = _props
        self._corenlp_client_args = _args
        self._threads = threads

    def annotate_with_custom_format(self, document: str) -> Iterator[Dict[str, Union[str, int, None, List[int]]]]:
        # annotate with serialized format
        obj_doc = self._corenlp[0].annotate(document, output_format="serialized")
        return parse_serialized_document(obj_doc)

    # 並列処理
    def annotate_with_custom_format_parallel(self, lst_documents: List[str])-> Iterator[Dict[str, Union[str, int, None, List[int]]]]:
        iter_clients = itertools.cycle(self._corenlp.values())
        iter_inputs = ((client, document) for client, document in zip(iter_clients, lst_documents))
        lst_docs = self._pool.starmap(_annotate_parse, iter_inputs)
        all_sentences = itertools.chain(*lst_docs)
        return all_sentences

    def annotate(self, document: str):
        return self._corenlp[0].annotate(document)


# exec stand-alone mode
if __name__ == "__main__":
    pass