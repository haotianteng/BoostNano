#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 06:56:41 2019

@author: heavens
"""

from __future__ import absolute_import
from __future__ import print_function
from boostnano import hmm
import numpy as np
from time import time

if __name__ == "__main__":
    segs = [10000,30000,40000,4000]
    probs = [[[0.6,0.1,0.2,0.1]],[[0.1,0.7,0.1,0.1]],[[0.1,0.1,0.8,0.0]],[[0.1,0.1,0.0,0.8]]]
    a = []
    for index,seg in enumerate(segs):
        a = a + probs[index]*seg
    a = np.asarray(a)
    start = time()
    (decoded,path,locs) = hmm.decode(a.transpose())
    end = time()
    print(decoded)
    print(path)
    print(locs)
    print(end-start)
