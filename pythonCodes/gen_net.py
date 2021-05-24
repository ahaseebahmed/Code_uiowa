#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 11:25:52 2020

@author: ahhmed
"""
import torch.nn as nn
import torch.nn.functional as F
#Define the generative model
class generator(nn.Module):
    # initializers
    def __init__(self):
        super(generator, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(10,15,4,2,1) # 1x1xsiz_latent->3x3x512
        self.deconv1_bn = nn.BatchNorm2d(15)
        self.deconv2 = nn.ConvTranspose2d(15, 20,4,2,1) # 3x3x512->5x5x256
        self.deconv2_bn = nn.BatchNorm2d(20)
        self.deconv3 = nn.ConvTranspose2d(20, 30, 4, 2, 1) # 5x5x256->10x10x128
        self.deconv3_bn = nn.BatchNorm2d(30)
        self.deconv4 = nn.ConvTranspose2d(30, 60, 4, 2, 1) # 10x10x128->20x20x128
        self.deconv4_bn = nn.BatchNorm2d(30)
        #self.deconv5 = nn.ConvTranspose2d(d, int(d/2), 3, 2, 0) # 20x20x128->41x41x64
        #self.deconv5_bn = nn.BatchNorm2d(int(d/2))
        #self.deconv6 = nn.ConvTranspose2d(int(d/2), int(d/4), 5, 2, 0) # 41x41x64->85x85x32
        #self.deconv6_bn = nn.BatchNorm2d(int(d/4))
        #self.deconv7 = nn.ConvTranspose2d(int(d/4), int(d/8), 4, 2, 1) # 85x85x32->170x170x16
        #self.deconv7_bn = nn.BatchNorm2d(int(d/8))
        #self.deconv8 = nn.ConvTranspose2d(int(d/8), out_channel, 4, 2, 1) # 170x170x16->340x340x2

    # forward method
    def forward(self, input):
        x = F.leaky_relu(self.deconv1_bn(self.deconv1(input)),0.2)
        x = F.leaky_relu(self.deconv2_bn(self.deconv2(x)),0.2)
        x = F.leaky_relu(self.deconv3_bn(self.deconv3(x)),0.2)
        x = self.deconv4(x)
        #x = F.leaky_relu(self.deconv5_bn(self.deconv5(x)),0.2)
        #x = F.leaky_relu(self.deconv6_bn(self.deconv6(x)),0.2)
        #x = F.leaky_relu(self.deconv7_bn(self.deconv7(x)),0.2)
        #x = F.leaky_relu(self.deconv8(x),0.2)
        return x