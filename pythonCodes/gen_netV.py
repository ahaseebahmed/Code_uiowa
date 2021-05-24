#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 11:25:52 2020

@author: ahhmed
"""
import torch.nn as nn
import torch.nn.functional as F
#Define the generative model
def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()
        
class generatorV(nn.Module):
    # initializers
    def __init__(self,N):
        super(generatorV, self).__init__()
        self.L1 = nn.Linear(N,N//2,False) 
        self.L2 = nn.Linear(N//2,N//4,False)
        self.L3 = nn.Linear(N//4,N//8,False)
        self.L4 = nn.Linear(N//8,30,False)

        
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)
    # forward method
    def forward(self, input):
        x = F.leaky_relu((self.L1(input)),0)
        x = F.leaky_relu((self.L2(x)),0)
        x = F.leaky_relu((self.L3(x)),0)
        x = self.L4(x)
        return x