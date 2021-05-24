#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 19:30:50 2020

@author: ahhmed
"""

import os,time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import scipy.io as sio
import h5py as h5
import random

#%%

for i in range(70,77):
    #print(i)
    plt.figure(i)
    plt.imshow(np.abs(data2[i,0,0]))
    
trnFiles=os.listdir('./../Codes/TensorFlow_SToRM/Data/')    
rn=[1,5,2,10,17,18,19,20,23,28,26,32,33,36,43,46,49,52,54,56,59,60,61,62,63,64,65,71,72,74]    
for fl in range(len(rn)):
    #matfile = sio.loadmat('./../Codes/TensorFlow_SToRM/Data/'+trnFiles[fl])
    print('./../Codes/TensorFlow_SToRM/Data/'+trnFiles[rn[fl]])