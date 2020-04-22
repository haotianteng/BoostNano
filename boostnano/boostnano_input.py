# Copyright HaotianTeng. All Rights Reserved.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Created on Tue May 7 04:16:40 2019
from __future__ import print_function, division
import os
import torch
import torch.nn as nn
import h5py
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
import torch.utils.data as data
from torchvision import transforms
import warnings
import csv
import uuid 

warnings.filterwarnings("ignore")

def _read_signal(file):
    try:
        with open(file,'r') as f:
            for line in f:
                split_line = line.strip().split()
                return str(uuid.uuid4()),[int(x) for x in split_line]
    except:
        return None,None

def _read_fast5(file):
    try:
        with h5py.File(file,mode='r') as root:
            signal_entry = list(root['/Raw/Reads'].values())[0]
            signal_attr=signal_entry.attrs
            read_id=signal_attr['read_id'].decode()
            raw_signal = np.asarray(signal_entry[('Signal')])
        return read_id,raw_signal[::-1]
    except OSError as e:
        return None,None

def read_csv(csv_file,root_dir):
    """
    Reading the signal according to the info on input csv file and extract the
    signals to the root_dir, the name of the files have been normalized.
    Args:
        csv_file: String, path to the input csv file that contain singal file 
            name and corresponding adapter segmentation.
        root_dir: String, path to the root directory of the data folder.
    Output:
        A List containing the singal-loc pair.
    """
    data = []
    E_STR=['success',
           'adapter segmentation failure',
           'fast5 file open error',
           'signal file open error']
    RUN_COUNT={E_STR[0]:0,
               E_STR[1]:0,
               E_STR[2]:0,
               E_STR[3]:0}
    if not os.path.isdir(root_dir):
        os.makedirs(root_dir,0o755)
    signal_folder = os.path.join(root_dir,'signals')
    signal_folder_idx = 0
    os.makedirs(signal_folder,0o755,exist_ok=True)
    out_csv = os.path.join(root_dir,'adapter_location.csv')
    out_f = open(out_csv,'w+')
    out_f.write("read_id, original_file, seg_0, seg_1, seg_2")
    with open(csv_file) as f:
        for line in tqdm(f):
            split_line = line.strip().split(',')
            if 'None' in split_line:
                RUN_COUNT[E_STR[1]]+=1
                continue
            file_n=split_line[0]
            if file_n.endswith('signal'):
                read_id,sig = _read_signal(file_n)
                if read_id is None:
                    RUN_COUNT[E_STR[3]]+=1
            elif file_n.endswith('fast5'):
                read_id,sig = _read_fast5(file_n)
                if read_id is None:
                    RUN_COUNT[E_STR[2]]+=1
                    continue
            sig = sig[::-1]
            sig_n = len(sig)
            locs = [sig_n-float(x) for x in split_line[1:]]
            out_f.write(','.join([read_id,file_n,str(locs[0]),str(locs[1]),str(locs[2])]))
            out_f.write('\n')
            with open(os.path.join(signal_folder, read_id),'w+') as sig_f:
                sig_f.write('original_file_path:'+file_n + '\n')
                sig_f.write(','.join([str(x) for x in sig]))
            RUN_COUNT[E_STR[0]]+=1
    return RUN_COUNT

def show_landmarks(sample):
    """
    Show the adapter segmentation landmarks of a sample.
    """
    plt.plot(sample['signal'],'grey')
    for i in range(3):    
        plt.axvline(sample['landmarks'][i],color = 'red')
        
def show_landmarks_batch(batch_sample):
    """
    Show the batch of the samples.
    """
    signal_batch, landmarks_batch = sample_batched['signal'],sample_batched['landmarks']
    batch_size = len(signal_batch)
    color = ['red','green','blue']
    for i in range(batch_size):
        plt.plot(signal_batch[i,:,0].numpy(),color = 'grey')
        for j in range(3):
            plt.axvline(x=landmarks_batch[i,j],color = color[j])

class dataset(data.Dataset):
    """
    Adapter segmentation dataset
    """
    def __init__(self, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.csv_file = os.path.join(root_dir,'adapter_location.csv')
        self.sig_folder = os.path.join(root_dir,'signals')
        self.transform = transform
        self.labels = []
        with open(self.csv_file,'r') as f:
            next(f)
            for line in tqdm(f):
                split_line = line.strip().split(',')
                locs = [float(x) for x in split_line[2:]]
                self.labels.append([split_line[0]]+locs)
                
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sig_f = os.path.join(self.sig_folder,
                                self.labels[idx][0])
        with open(sig_f,'r') as f:
            next(f)
            for line in f:
                sig = line.strip().split(',')
                sig = np.asarray([int(x) for x in sig],dtype=np.float32)
        sample = {'signal': sig, 
                  'landmarks': np.asarray(self.labels[idx][1:4],dtype=np.float32)}
        if self.transform:
            sample = self.transform(sample)
        return sample

class DeNoise(object):
    """Remove the signal point that is exceed a given region.

    Args:
        Min and Max (tuple of 2): The minimum and maximum signal point that is allowed.
    """
    def __init__(self, minmax):
        assert isinstance(minmax, (tuple))
        assert len(minmax)==2
        self.min_sig=minmax[0]
        self.max_sig=minmax[1]
        
    def __call__(self, sample):
        sig,landmarks = sample['signal'],sample['landmarks']
        sig[sig>self.max_sig] = self.max_sig
        sig[sig<self.min_sig] = self.min_sig
        return {'signal':sig,'landmarks':landmarks}

class WhiteNoise(object):
    """Add white noise to the signal.
    Args:
        sampling_region(int or Tuple): Sampling region. If tuple, then sampling
        the white noise from the [tuple[0],tuple[1]] segment, if int, region 
        are [0,tuple], first get the std of the region and then generate white
        noise with the same std.
        amplitude(float): The amplitude of the sampling standard error applying
        to the white noise.
    """
    def __init__(self,sampling_region,amplitude):
        assert isinstance(sampling_region,(int,tuple))
        assert isinstance(amplitude,float)
        self.sr=sampling_region
        self.amp=amplitude
    def __call__(self,sample):
        sig,landmarks = sample['signal'],sample['landmarks']
        if isinstance(self.sr, int):
            std = np.std(sig[:self.sr])
            white_noise = np.random.normal(0, std, size=len(sig))
        else:
            std = np.std(sig[self.sr[0]:self.sr[1]])
            white_noise = np.random.normal(0, std, size=len(sig))
        return {'signal':np.asarray(sig+self.amp*white_noise,dtype=np.float32),'landmarks':landmarks}            
class Crop(object):
    """Crop(padding) the signal into a certain length.
    Args:
        length(Int): The desired length.
        
    """
    def __init__(self,length):
        assert isinstance(length,int)
        self.length=length
    def __call__(self,sample):
        sig,landmarks = sample['signal'],sample['landmarks']
        sig_n = len(sig)
        if sig_n < self.length:
            sig = np.pad(sig, (0,self.length - sig_n), 'constant', constant_values = (0,0) )
        else:
            sig = sig[:self.length]
            sig_n = self.length
        mask = np.zeros(self.length)
        mask[:sig_n] = 1
        return {'signal':sig,'landmarks':landmarks,
                'signal_length':np.asarray(sig_n,dtype = np.int16),
                'signal_mask':np.asarray(mask,dtype=np.float32)}

class TransferProb(object):
    """Calculate the transfer probability using gaussian distribution of the distance
    from the given landmarks. P=exp(-(p-dist)^2/(Theta^2))
    Args:
        Theta: The standard error of the gaussian distribution.
    """
    def __init__(self,theta):
        self.theta = theta
    def __call__(self,sample):
        signal,landmarks,length = sample['signal'],sample['landmarks'],sample['signal_length']
        sig_n = len(signal)
        idx=np.arange(sig_n)
        tp = np.zeros((3,sig_n),dtype = np.float32)
        c = np.zeros((sig_n),dtype = np.int64)
        locs = [int(x) for x in landmarks]
        c[:locs[0]] = 0
        c[locs[0]:locs[1]] = 1
        c[locs[1]:locs[2]] = 2
        c[locs[2]:length] = 3
        for loc_i,loc in enumerate(landmarks):
            tp[loc_i,:]=np.exp(-(idx - loc)**2/(self.theta**2))
            tp[:,length:] = 0
        return {'signal':signal,
                'signal_length':length,
                'signal_mask':sample['signal_mask'],
                'landmarks':landmarks,
                'transfer':tp,
                'class':c}
        
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    
    def __call__(self, sample):
        signal = sample['signal']
        sig_n = len(signal)
        signal = np.reshape(signal,[sig_n,1])
        return {'signal': torch.from_numpy(signal),
                'signal_length':torch.from_numpy(sample['signal_length']),
                'signal_mask':torch.from_numpy(sample['signal_mask']),
                'landmarks': torch.from_numpy(sample['landmarks']),
                'transfer': torch.from_numpy(sample['transfer']),
                'class':torch.from_numpy(sample['class'],)}

if __name__ == "__main__":
    root_dir = '/home/heavens/UQ/Chiron_project/boostnano/training_data/test/'
    test_file = "/home/heavens/UQ/Chiron_project/boostnano/training_data/training.csv"
#    data = read_csv("/home/heavens/UQ/Chiron_project/boostnano/training_data/result_soeyoung.csv",'/home/heavens/UQ/Chiron_project/boostnano/training_data/eval')
#    data = read_csv(test_file,root_dir)
    d1 = dataset(root_dir,transform=transforms.Compose([DeNoise((200,1200)),
                                                        WhiteNoise(200,0.1),
                                                        Crop(30000),
                                                        TransferProb(5),
                                                        ToTensor()]))
    dataloader = data.DataLoader(d1,batch_size=1,shuffle=True,num_workers=4)
    count = [0,0,0]
    for i_batch, sample_batched in enumerate(dataloader):
#        print(sample_batched['signal'].size())
#        print(sample_batched['signal_length'].size())
#        print(sample_batched['landmarks'].size())
#        print(sample_batched['transfer'].size())
#        print(sample_batched['class'].size())
        if i_batch==2:
            show_landmarks_batch(sample_batched)
            break
        for i in range(3):
            temp_count = sample_batched['class'] == i
            temp_count = temp_count.numpy().astype(np.int32)
            count[i] += sum(sum(temp_count))
        
        
        
    