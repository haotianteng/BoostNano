#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 13:17:41 2019

@author: heavens
"""
from nanopre.nanopre import CSM
from nanopre import hmm
import argparse

class evaluater(object):
    def __init__(self,net):
        self.net = net
        
        
    def eval_once(self,signal,segment_len):
        pass
     
    def decoding(self):
        hmm.decode(self.logits)

def test():
    print(hmm.decode(1,2))

if __name__ == "__main__":
    test_model = ""
#    parser = argparse.ArgumentParser(prog='chiron',
#                                     description='A deep neural network basecaller.')
#    parser.add_argument('-i', 
#                        '--input_fsat5_folder', 
#                        required = True,
#                        help="File path or Folder path to the fast5 file.")

    