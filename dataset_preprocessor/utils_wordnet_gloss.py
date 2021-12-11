#!/usr/bin/env python
# -*- coding:utf-8 -*-

def clean_up_lemma(str_lemma: str):
    if str_lemma.find("|") != -1:
        str_lemma = str_lemma.split("|")[0]
    if str_lemma.find("%") != -1:
        str_lemma = str_lemma.split("%")[0]
    return str_lemma

def clean_up_surface(str_surface: str):
    return str_surface.replace("\n", "")

def lemma_to_surface(str_lemma: str):
    str_lemma = clean_up_lemma(str_lemma)
    str_lemma = str_lemma.replace("_", " ")
    return str_lemma