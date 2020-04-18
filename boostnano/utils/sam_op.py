#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 21:18:30 2020

@author: heavens
"""
import re
import pickle
COMPLEMENT = {"A":"T",
              "C":"G",
              "G":"C",
              "T":"A",
              "a":"T",
              "c":"G",
              "g":"C",
              "t":"A",
              "U":"A",
              "u":"A"}
TAB = ''.maketrans(COMPLEMENT)
def save_aligner(aln,save_f):
    with open(save_f,'wb+') as f:
        pickle.dump(aln,f)

def load_aligner(save_f):
    with open(save_f,'rb') as f:
        aln = pickle.load(f)
    return aln

def reverse_complement(sequence):
    return sequence[::-1].translate(TAB)

class Aligner(object):
    def __init__(self,ref_file):
        self.aln_dict = {}
        self.ref = {}
        self._parse_ref(ref_file)
    
    def parse_sam(self,samfile):
        with open(samfile,'r') as f:
            for line in f:
                if line.startswith("@"):
                    continue
                self._parse_line(line)
                
    def _parse_line(self,line):
        line = re.split(r'[\t\s]\s*', line.strip())
        if line[0] == '':
            line = line[1:]
        read_id = line[0]
        ref_contig = line[2]
        if ref_contig == "*":
            return
        ref_start = int(line[3])-1
        flag = int(line[1])
        rc = bool(flag&0X10)
        cigar_string = line[5]
        map_score = int(line[4])
        cigar_split = re.split("([A-Z])",cigar_string)[:-1]
        cigar_iter = iter(cigar_split)
        query = line[9]
        if rc:
            query = reverse_complement(query)
        query_offset =[0,len(query)]
        if cigar_split[1] == 'S':
            if rc:
                query_offset[1] = len(query) - int(cigar_split[0])
            else:
                query_offset[0] = int(cigar_split[0])
        if cigar_split[-1] == 'S':
            if rc:
                query_offset[0] = int(cigar_split[-2])
            else:
                query_offset[1] = len(query) - int(cigar_split[-2])
        def increment_char(c):
            return c=='M' or c=='D' or c=='N'
        ref_len = sum([int(x) for x in cigar_iter if increment_char(next(cigar_iter)) ])
        ref_seq = self.ref[ref_contig][ref_start:ref_start+ref_len]
        if rc:
            ref_seq = reverse_complement(ref_seq)
        if read_id in self.aln_dict.keys():
            self.aln_dict[read_id].append({"map_score":map_score,
                                           "reference_sequence":ref_seq,
                                           "query_sequence":query,
                                           "query_offset":query_offset})
        else:
            self.aln_dict[read_id] = [{"map_score":map_score,
                                       "reference_sequence":ref_seq,
                                       "query_sequence":query,
                                       "query_offset":query_offset}]
            
        
    def _parse_ref(self,ref_f):
        contig_id = None
        current_seq = ''
        with open(ref_f,'r') as f:
            for line in f:
                if line.startswith(">") or line.startswith("@"):
                    if contig_id is not None:
                        self.ref[contig_id] = current_seq
                    contig_id = line.strip()[1:]
                    current_seq = ''
                else:
                    current_seq = current_seq + line.strip()
            self.ref[contig_id] = current_seq

if __name__ == "__main__":
    #Example usage
    aln = Aligner("/home/heavens/UQ/Chiron_project/RNA_Analysis/RNA_Nanopore/references/sythetic.fasta")
    aln.parse_sam("/home/heavens/UQ/Chiron_project/RNA_Analysis/RNA_Nanopore/assess/out.sam")
    save_aligner(aln,"/home/heavens/UQ/Chiron_project/RNA_Analysis/RNA_Nanopore/assess/aln.bin")
    aln2 = load_aligner("/home/heavens/UQ/Chiron_project/RNA_Analysis/RNA_Nanopore/assess/aln.bin")