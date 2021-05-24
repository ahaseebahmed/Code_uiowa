#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 11:25:52 2020

@author: ahhmed
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
#Define the generative model
       
class generator(nn.Module):
    # initializers
    def __init__(self):
        super(generator, self).__init__()
        self.L1 = nn.Linear(600,1200) 
        self.deconv1 = nn.ConvTranspose2d(1,64,(4,3),(2,1),(1,1))#input=1 output=1 kernel=4x5 stride=2x1 padding =1
        #self.deconv11 = nn.ConvTranspose2d(128,64,(1,3),(3,1),(0,1)) 
        self.deconv2 = nn.ConvTranspose2d(128,64,(4,3),(2,1),(1,1))
        #self.deconv21 = nn.ConvTranspose2d(1,1,(3,5),(1,1),(1,2)) 
        self.deconv3 = nn.ConvTranspose2d(64,32,(4,3),(2,1),(1,1))
        #self.deconv31 = nn.ConvTranspose2d(1,1,(3,5),(1,1),(1,2)) 
        #self.deconv4 = nn.ConvTranspose2d(1,1,(4,5),(2,1),(1,2))
        #self.deconv41 = nn.ConvTranspose2d(1,1,(3,5),(1,1),(1,2)) 
        self.deconv5 = nn.ConvTranspose2d(32,1,(4,3),(2,1),(1,1)) 

    # forward method
    def forward(self, input):
        x=self.L1(input)
        x=torch.reshape(x,(1,1,4,300))
        x = F.leaky_relu(self.deconv1(x),0.2)
        #x = F.leaky_relu(self.deconv11(x),0)
        #x = F.leaky_relu(self.deconv2(x),0.2)
        #x = F.leaky_relu(self.deconv21(x),0)
        x = F.leaky_relu(self.deconv3(x),0.2)
        #x = F.leaky_relu(self.deconv31(x),0)
        #x = F.leaky_relu(self.deconv4(x),0)
        #x = F.leaky_relu(self.deconv41(x),0)
        x = self.deconv5(x)
        return x