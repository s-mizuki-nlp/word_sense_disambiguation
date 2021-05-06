#!/usr/bin/env python
# -*- coding:utf-8 -*-

from torchtext.datasets.wikitext103 import _RawTextIterableDataset

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