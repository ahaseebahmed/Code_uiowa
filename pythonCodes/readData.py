"""
Created on Tue Mar 24 12:15:06 2020

@author: haggarwal
"""


import h5py as h5
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class DataGen(Dataset):
    def __init__(self,inp,org):

#        org=torch.tensor(org)
#        inp=torch.tensor(inp)
#        org=org.unfold(2,32,32).unfold(3,128,128)
#        inp=inp.unfold(2,32,32).unfold(3,128,128)
#        inp=inp.permute(0,2,3,1,4,5)
#        org=org.permute(0,2,3,1,4,5)
#        inp=inp.reshape(inp.shape[0]*inp.shape[1]*inp.shape[2],inp.shape[3],inp.shape[4],inp.shape[5])
#        org=org.reshape(org.shape[0]*org.shape[1]*org.shape[2],org.shape[3],org.shape[4],org.shape[5])

        self.nImg=inp.shape[0]
        self.inp=inp
        self.org=org

    def __len__(self):
        return self.nImg


    def __getitem__(self, idx):
        inp=self.inp[idx]
        org=self.org[idx]
        inp=torch.tensor(inp)
        org=torch.tensor(org)
        #inp.requires_grad=True
        return inp,org#,csm,mask

