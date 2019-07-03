#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 15:27:09 2019

@author: heavens
"""
import h5py
from torchvision import transforms
from torch.utils import data
import torch
import torch.nn as nn
import numpy as np

class CSM(nn.Module):
    """
    The convolutional segmentation machine.
    
    """
    def __init__(self):
        super(CSM, self).__init__()
        self.softmax = nn.LogSoftmax(dim=1)
        self.CrossEntropyLoss = nn.CrossEntropyLoss(reduction='none')
        self.main_branch = []
        self.conv0_0 = nn.Conv1d(1,32,5,stride=1, padding=2)
        self.main_branch.append(self.conv0_0)
        self.relu0_0 = nn.LeakyReLU()
        self.main_branch.append(self.relu0_0)
        self.conv0_1 = nn.Conv1d(32,64,3,stride=1,padding=1)
        self.main_branch.append(self.conv0_1)
        self.relu0_1 = nn.LeakyReLU()
        self.main_branch.append(self.relu0_1)
#        self.conv0_2 = nn.Conv1d(64,64,3,stride=1,padding=1)
#        self.main_branch.append(self.conv0_2)
#        self.relu0_2 = nn.LeakyReLU()
#        self.main_branch.append(self.relu0_2)
        
        self.branch0 = []
        self.conv1_0 = nn.Conv1d(64,64,3,stride=1,padding=1)
        self.branch0.append(self.conv1_0)
        self.relu1_0 = nn.LeakyReLU()
        self.branch0.append(self.relu1_0)
        self.conv1_1 = nn.Conv1d(64,128,3,stride=1,padding=1)
        self.branch0.append(self.conv1_1)
        self.relu1_1 = nn.LeakyReLU()
        self.branch0.append(self.relu1_1)
        self.conv1_2 = nn.Conv1d(128,32,1,stride=1,padding=0)
        self.branch0.append(self.conv1_2)
        self.relu1_2 = nn.LeakyReLU()
        self.branch0.append(self.relu1_2)
        self.c2o_0 = nn.Conv1d(32,4,1,stride=1,padding=0)
        
        self.branch1 = []
        self.conv1 = nn.Conv1d(96,128,5,stride=1,padding=2)
        self.branch1.append(self.conv1)
        self.relu1 = nn.LeakyReLU()
        self.branch1.append(self.relu1)
        self.conv2 = nn.Conv1d(128,256,5,stride=1,padding=2)
        self.branch1.append(self.conv2)
        self.relu2 = nn.LeakyReLU()
        self.branch1.append(self.relu2)
        self.conv3 = nn.Conv1d(256,32,1,stride=1,padding=0)
        self.branch1.append(self.conv3)
        self.relu3 = nn.LeakyReLU()
        self.branch1.append(self.relu3)
        self.c2o = nn.Conv1d(32,4,1,stride=1,padding=0)
    
    def forward_0(self,feature):
        """The forward network for the first segment input.
        Args:
            Feature: A batch of the signal has the shape of [Batch_size,1,Segment_length]
        Output:
            Feature: Output feature (hidden unit) with the shape [Batch_size,32,Segment_length] 
            Out: Classification probability with the shape [Batch_size,4,Segment_length]
        """
        for layer in self.main_branch:
            feature = layer(feature)
        for layer in self.branch0:
            feature = layer(feature)
        out = self.c2o_0(feature)
        return feature,out
        
    def forward(self, input, hidden):
        feature = input
        for layer in self.main_branch:
            feature = layer(feature)
        combined = torch.cat((feature,hidden),1)
        for layer in self.branch1:
            combined = layer(combined)
        out = self.c2o(combined)
        return combined,out
    
    def celoss(self, predict, target,mask):
        loss = self.CrossEntropyLoss(predict,target)
        return loss * mask
    
    def error(self, predict, target, mask):
        mask = mask.numpy()
        predict = predict.argmax(1)
        compare = (predict == target)
        compare = compare.numpy()
        error = (1 - compare)*mask
        error = np.sum(error,1) / (np.sum(mask,1)+1e-5)
        error = np.mean(error)
        return error
