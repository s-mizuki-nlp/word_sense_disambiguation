#!/usr/bin/env python
# -*- coding:utf-8 -*-

from typing import Sequence
import os, sys, io

def pick_first_available_path(*lst_path):
    for path in lst_path:
        if os.path.exists(path):
            return path
    return path
    # raise FileNotFoundError(f"all path failed: {lst_path}")