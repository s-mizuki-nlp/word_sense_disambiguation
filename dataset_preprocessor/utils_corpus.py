#!/usr/bin/env python
# -*- coding:utf-8 -*-

from torchtext.datasets.wikitext103 import _RawTextIterableDataset
from datasets.arrow_dataset import Dataset
import unicodedata

from .utils import strip_tags

def batch_document_generator_for_wikitext_dataset(wikitext_dataset: _RawTextIterableDataset, batchsize: int):
    txt = ""; idx = 0
    for doc in wikitext_dataset:
        doc = doc.strip()
        if len(doc) == 0:
            continue
        # skip article title
        if doc.startswith("=") and doc.endswith("="):
            continue
        txt += doc
        idx += 1

        if idx >= batchsize:
            yield txt
            txt = ""
            idx = 0


def batch_document_generator_for_wiki40b_dataset(wiki40b_dataset: Dataset, batchsize: int, first_paragraph_only: bool,
                                                 nfkc_normalize: bool = True, strip_xml_tag: bool = True):
    txt = ""; idx = 0
    for article in wiki40b_dataset:
        if first_paragraph_only:
            doc = _extract_first_paragraph_from_wiki40b_text(article["text"])
        else:
            doc = _extract_paragraph_from_wiki40b_text(article["text"])
        if len(doc) <= 10:
            continue

        txt += doc; txt += "\n"
        idx += 1

        if idx >= batchsize:
            # remove invalid unicode characters with NFKC normalization
            if nfkc_normalize:
                txt = unicodedata.normalize("NFKC", txt)
            # remove xml tags to avoid corenlp analysis error.
            if strip_xml_tag:
                txt = strip_tags(txt)

            yield txt.strip("\n")
            txt = ""
            idx = 0


def _extract_paragraph_from_wiki40b_text(str_wiki40b_text: str, newline: str = "\n", delimiter: str = "\n"):
    # strip first/last newlines
    txt = str_wiki40b_text.strip()

    # split text into sections
    lst_texts = txt.split("\n")

    # extract paragraphs
    it_marker_text_tuple = zip(lst_texts[0::2], lst_texts[1::2])
    it_ret = (text for marker, text in it_marker_text_tuple if marker == "_START_PARAGRAPH_")

    # concat and replace newline symbol
    return delimiter.join(it_ret).replace("_NEWLINE_", newline)

def _extract_first_paragraph_from_wiki40b_text(str_wiki40b_text: str, newline: str = "\n"):
    # strip first/last newlines
    txt = str_wiki40b_text.strip()

    # split text into sections
    lst_texts = txt.split("\n")

    # extract first paragraph
    for idx, text in enumerate(lst_texts):
        # if article begins with section, it indicates there is not first paragraph. thus returns empty string.
        if text == "_START_SECTION_":
            return ""
        if text == "_START_PARAGRAPH_":
            break

    if idx == len(lst_texts) - 1:
        return ""

    first_paragraph = lst_texts[idx+1]
    # replace newline symbol
    return first_paragraph.replace("_NEWLINE_", newline)
