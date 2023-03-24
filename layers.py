#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 16:17:09 2022

@author: apramanik
"""


import torch.nn as nn
import utils.spectral_norm_chen as chen


class convlayer(nn.Module):
    
    def __init__(self, input_channels, output_channels, last, sn=False):
        super(convlayer, self).__init__()
        
        
        if sn:
            self.conv = chen.spectral_norm(nn.Conv2d(in_channels=input_channels,out_channels=output_channels,kernel_size=3,stride=1,padding=1,bias=True))
            
        else:
            self.conv = nn.Conv2d(in_channels=input_channels,out_channels=output_channels,kernel_size=3,stride=1,padding=1,bias=True)
        
        
        self.relu = nn.ReLU()
        self.last = last
        
    def forward(self,x):
        x = self.conv(x)
        if not self.last:
            x = self.relu(x)
        return x