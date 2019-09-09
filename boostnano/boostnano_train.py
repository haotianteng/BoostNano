#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 12 00:47:58 2019

@author: heavens
"""

import boostnano.boostnano_input as ni
from torchvision import transforms
from torch.utils import data
import torch
import torch.nn as nn
import numpy as np
import os 
from boostnano.boostnano_model import CSM

class trainer(object):
    def __init__(self,segment_len,train_dataloader,net,keep_record = 5,eval_dataloader = None,device = None):
        """Trainer
        Args:
            segment_len: Type Int, legnth fo the segment.
            train_dataloader: A torch.utils.data.dataloader.DataLoader instance.
            net: A CSM instance.
            keep_record: Type Int, defualt is 5, the latest n checkpoints to save for each training routine.
            eval_dataloader: A torch.utils.data.dataloader.DataLoader instance, if None, use training dataloader.
            device: The device to put the model on, can be cpu or cuda, if None then gpu will be used if available.
        """
        self.segment_len = segment_len
        self.train_ds = dataloader
        self.device = self._get_device(device)
        if eval_dataloader is None:
            self.eval_ds = self.train_ds
        else:
            self.eval_ds = eval_dataloader
        self.net = net
        self.global_step = 0
        self.keep_record = keep_record
        self.save_list = []
    
    def _get_device(self,device):
        if device is None:
            if torch.cuda.is_available():
                return torch.device('cuda')
            else:
                return torch.device('cpu')
        else:
            return torch.device(device)
        
    def save(self):
        ckpt_file = os.path.join(self.save_folder,'checkpoint')
        current_ckpt = 'ckpt-'+str(self.global_step)
        model_file = os.path.join(self.save_folder,current_ckpt)
        self.save_list.append(current_ckpt)
        if not os.path.isdir(self.save_folder):
            os.mkdir(self.save_folder)
        if len(self.save_list) > self.keep_record:
            os.remove(os.path.join(self.save_folder,self.save_list[0]))
            self.save_list = self.save_list[1:]
        with open(ckpt_file,'w+') as f:
            f.write("latest checkpoint:" + current_ckpt + '\n')
            for path in self.save_list:
                f.write("checkpoint file:" + path + '\n')
        torch.save(self.net.state_dict(),model_file)
    
    def load(self,save_folder):
        self.save_folder = save_folder
        ckpt_file = os.path.join(self.save_folder,'checkpoint')
        with open(ckpt_file,'r') as f:
            latest_ckpt = f.readline().strip().split(':')[1]
            self.global_step = int(latest_ckpt.split('-')[1])
        self.net.load_state_dict(torch.load(os.path.join(save_folder,latest_ckpt),map_location=self.device))
        self.net.to(self.device)
        
    def train(self,epoches,optimizer,save_cycle,save_folder):
        self.save_folder = save_folder
        for epoch_i in range(epoches):
            for i_batch, batch in enumerate(self.train_ds):
                if i_batch%save_cycle==0:
                    calculate_error = True
                else:
                    calculate_error = False
                loss,error = self.train_step(batch,get_error = calculate_error)
                optimizer.zero_grad()
                loss.backward()
                if i_batch%save_cycle==0:
                    self.save()
                    eval_i,valid_batch = next(enumerate(self.eval_ds))
                    valid_error = self.valid_step(valid_batch)
                    print(valid_error)
                    print("Epoch %d Batch %d, loss %f, error %f, valid_error %f"%(epoch_i, i_batch, loss,np.mean(error),np.mean(valid_error)))
                optimizer.step()
                self.global_step +=1
                
    def valid_step(self,batch):
        signal_batch = batch['signal'].transpose(1,2)
        batch_shape = signal_batch.shape
        hidden,out = net.forward(signal_batch[:,:,:self.segment_len],np.asarray([]))
        label = batch['class'][:,:self.segment_len]
        mask = batch['signal_mask']
        error = []
        error.append(net.error(out,label,mask[:,:self.segment_len]))
        for i in range(self.segment_len,batch_shape[2],self.segment_len):
            hidden,out = net.forward(signal_batch[:,:,i:i+self.segment_len],hidden)
            label = batch['class'][:,i:i+self.segment_len]
            error.append(net.error(out,label,mask[:,i:i+self.segment_len]))
        return np.asarray(error)

    def train_step(self,batch,get_error = False):
        signal_batch = batch['signal'].transpose(1,2)
        batch_shape = signal_batch.shape
        hidden,out = net.forward(signal_batch[:,:,:self.segment_len],np.asarray([]))
        label = batch['class'][:,:self.segment_len]
        mask = batch['signal_mask']
        loss = net.celoss(out,label,mask[:,:self.segment_len])
        error = None
        if get_error:
            error = []
            error.append(net.error(out,label,mask[:,:self.segment_len]))
        for i in range(3000,batch_shape[2],self.segment_len):
            hidden,out = net.forward(signal_batch[:,:,i:i+self.segment_len],hidden)
            label = batch['class'][:,i:i+self.segment_len]
            loss += net.celoss(out,label,mask[:,i:i+self.segment_len])
            if get_error:
                error.append(net.error(out,label,mask[:,i:i+self.segment_len]))
        loss = torch.mean(loss)
        return loss,np.asarray(error)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dataloader, device = None):
        self.dataloader = dataloader
        if device is None:
            device = self.get_default_device()
        else:
            device = torch.device(device)
        self.device = device
    
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dataloader:
            yield self._to_device(b, self.device)
    
    def __len__(self):
        """Number of batches"""
        return len(self.dataloader)
    
    def _to_device(self,data,device):
        if isinstance(data, (list,tuple)):
            return [self._to_device(x,device) for x in data]
        if isinstance(data, (dict)):
            temp_dict = {}
            for key in data.keys():
                temp_dict[key] = self._to_device(data[key],device)
            return temp_dict
        return data.to(device, non_blocking=True)
    
    def get_default_device(self):
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')

if __name__ == "__main__":
    train_dir = '/home/heavens/UQ/Chiron_project/boostnano/training_data/test/'
    eval_dir = '/home/heavens/UQ/Chiron_project/boostnano/training_data/eval/'
    test_file = "/home/heavens/UQ/Chiron_project/boostnano/training_data/training.csv"
#    data = read_csv(test_file,root_dir)
    d1 = ni.dataset(train_dir,transform=transforms.Compose([ni.DeNoise((0,900)),
                                                        ni.WhiteNoise(200,0.2),
                                                        ni.Crop(30000),
                                                        ni.TransferProb(5),
                                                        ni.ToTensor()]))
    d2 = ni.dataset(eval_dir,transform=transforms.Compose([ni.DeNoise((0,900)),
                                                        ni.WhiteNoise(200,0.2),
                                                        ni.Crop(30000),
                                                        ni.TransferProb(5),
                                                        ni.ToTensor()]))
    device = "cuda"
    dataloader = DeviceDataLoader(data.DataLoader(d1,batch_size=5,shuffle=True,num_workers=5),device = device)
    eval_dataloader = DeviceDataLoader(data.DataLoader(d2,batch_size=5,shuffle=True,num_workers=5),device = device)
    net = CSM()
    t = trainer(1000,dataloader, net,eval_dataloader = eval_dataloader,device = device)
    lr = 1e-5
    epoches = 10
    global_step = 0
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
    COUNT_CYCLE = 10
    t.load("/home/heavens/UQ/Chiron_project/boostnano/training_data/test/model/")
    t.train(epoches,optimizer,COUNT_CYCLE,"/home/heavens/UQ/Chiron_project/boostnano/training_data/test/model/")

    
    
