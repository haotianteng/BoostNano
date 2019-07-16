#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 13:17:41 2019

@author: heavens
"""
from nanopre.nanopre_model import CSM
from nanopre import hmm
import torch
import os
import sys
import numpy as np
import h5py
import argparse
from tqdm import tqdm
from scipy.special import softmax
from matplotlib import pyplot as plt

class evaluator(object):
    def __init__(self,net,saved_folder,device=None):
        """Initialize a evaluator with a saved pytorch model
        Args:
            net: A pytorch model.
            saved_folder: the folder that contained a trained pytorch model.
            device: The running device, can be assign one of the following: cpu, cuda. If None, a device will be detected.
        """
        self.net = net
        self.save_folder = saved_folder
        self.device = self._get_device(device)
        ckpt_file = os.path.join(self.save_folder,'checkpoint')
        with open(ckpt_file,'r') as f:
            latest_ckpt = f.readline().strip().split(':')[1]
            self.global_step = int(latest_ckpt.split('-')[1])
        self.net.load_state_dict(torch.load(os.path.join(self.save_folder,latest_ckpt),
                                            map_location=self.device))
        self.net.to(self.device)
        
    def init_session(self,sample):
        #TODO: transfer the model into a traced model.
        """Transfer the saved model into a traced model:
        Args:
            sample: A sample input with shape of [Batch_size,1,segment_len]
        """
        self.net = torch.jit.trace(self.net,sample)
    
    def _get_device(self,device):
        if device is None:
            if torch.cuda.is_available():
                return torch.device('cuda')
            else:
                return torch.device('cpu')
        else:
            return torch.device(device)

    
    def eval_sig(self,signal,segment_len):
        """Evalulate a single signal.
        Args:
            signal: A 1-D signal.
            segment_len: Integer that indicate the segment length.
        Return:
            A tuple of (decoded,path,locs), decoded is the 
        """
        hidden = np.asarray([])
        sig_len = len(signal)
#        signal = np.reshape(signal,(1,1,len(signal)))
        out = np.reshape(np.asarray([[[]]]),(1,4,0))
        for i in range(0,sig_len,segment_len):
            segment = signal[i:i+segment_len]
            if len(segment)<segment_len:
                segment = np.pad(segment, (0,segment_len - len(segment)), 'constant', constant_values = (0,0) )
            segment = torch.from_numpy(np.reshape(segment,(1,1,segment_len)))
            with torch.no_grad():
                hidden,temp_out = self.net.forward(segment.to(self.device, non_blocking=True),hidden)
            out = np.concatenate((out,temp_out.detach().cpu().numpy()),axis = 2)
            torch.cuda.empty_cache()
        out = softmax(out, axis=1)
        return self.decoding(out[0,:,:])
            
    def decoding(self,logits):
        result_tuple = hmm.decode(logits)
        return result_tuple
    
def fast5_iter(fast5_dir):
    for (dirpath, dirnames, filenames) in os.walk(fast5_dir+'/'):
        for filename in filenames:
            if not filename.endswith('fast5'):
                continue
            abs_path = os.path.join(dirpath,filename)
            root = h5py.File(abs_path)
            signal = np.asarray(list(root['/Raw/Reads'].values())[0][('Signal')],dtype = np.float32)
            yield signal,abs_path
    
def trace(net, example_input):
    traced_model = torch.jit.trace(net,example_input)
    return traced_model
    
def main():
    net = CSM()
    ev = evaluator(net,FLAGS.model_path,device = FLAGS.device)
    iterator = fast5_iter(FLAGS.input_fast5)
    if not os.path.isdir(FLAGS.output_folder):
        os.mkdir(FLAGS.output_folder)
    output_f = os.path.join(FLAGS.output_folder, 'out.csv')
    with open(output_f,'w+') as out_f:
        for signal,fast5_f in tqdm(iterator):
            (decoded,path,locs) = ev.eval_sig(signal,FLAGS.segment_length)
            out_f.write(','.join([fast5_f]+[str(x) for x in locs])+'\n')
    
def test(fast5_f):
    root = h5py.File(fast5_f)
    signal = np.asarray(list(root['/Raw/Reads'].values())[0][('Signal')],dtype = np.float32)
    net = CSM()
    ev = evaluator(net,FLAGS.model_path,device = FLAGS.device)
    (decoded,path,locs) = ev.eval_sig(signal,FLAGS.segment_length)
    print(locs)
    plt.plot(signal)
    for loc in locs:
        plt.axvline(x=loc,color = 'red')
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='chiron',
                                     description='A deep neural network basecaller.')
    parser.add_argument('-i', 
                        '--input_fast5', 
                        required = True,
                        help="File path or Folder path to the fast5 file.")
    parser.add_argument('-m', 
                        '--model_path', 
                        required = True,
                        help="Folder that contain the model.")
    parser.add_argument('-o',
                        '--output_folder',
                        required = True,
                        help="Folder to write result.")
    parser.add_argument('-r',
                        '--recursive', 
                        action='store_true',
                        help="If read the files recursively.")
    parser.add_argument('-l',
                        '--segment_length',
                        default = 1000,
                        type = int,
                        help="Folder to write result.")
    parser.add_argument('-t',
                        '--threads',
                        default = 1,
                        type = int,
                        help="Number of threads that are used to run.")
    parser.add_argument('-d',
                        '--device', 
                        default = None,
                        help="Calculation device, need to be one of the following: cpu, cuda:0, cuda:1, ...")
    ##TODO Multiple threading need to be added.
    FLAGS = parser.parse_args(sys.argv[1:])
    if (FLAGS.device != "cpu") and (FLAGS.device is not None) and (not FLAGS.device.startswith("cuda:")):
        raise ValueError("Invalid device %s"%(FLAGS.device))
    main()
    
     
    
    