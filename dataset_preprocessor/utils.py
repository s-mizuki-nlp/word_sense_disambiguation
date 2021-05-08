#!/usr/bin/env python
# -*- coding:utf-8 -*-

from io import StringIO
from html.parser import HTMLParser

class MLStripper(HTMLParser):

    def __init__(self, strict: bool = False, convert_charrefs: bool = True):
        super().__init__()
        self.reset()
        self.strict = strict
        self.convert_charrefs = convert_charrefs
        self.text = StringIO()

    def handle_data(self, d):
        self.text.write(d)

    def get_data(self):
        return self.text.getvalue()

def strip_tags(html):
    s = MLStripper()
    s.feed(html)

    return s.get_data()