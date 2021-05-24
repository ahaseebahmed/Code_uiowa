#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 15:04:07 2020

@author: ahhmed
"""
import os,time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import gen_net as gn
import gen_net3 as gn3
import scipy.io as sio
import random

import supportingFun as sf
from torch.autograd import Variable
import h5py as h5
import readData as rd
import scipy.io as sio
import sys
from torch.utils.data import TensorDataset

sys.path.insert(0, "/Users/ahhmed/pytorch_unet")

from torchkbnufft.torchkbnufft import KbNufft
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

#from model import UnetClass
from spatial_model6 import SmallModel
from unetmodel import UnetClass



from scipy import ndimage
import gen_net1 as gn1
gpu=torch.device('cuda:0')
cpu=torch.device('cpu')
#directory='/Users/ahhmed/Codes/TensorFlow_SToRM/Data1/FID142454v/d900/'
directory='/Users/ahhmed/Codes/TensorFlow_SToRM/tstData/storm_900/'

dtype = torch.float
directory1='/Users/ahhmed/pytorch_unet/SPTmodels/19Jun_101938pm_50ep_20Jun/'
#directory1='/Users/ahhmed/pytorch_unet/SPTmodels/21Jun_110930pm_100ep_21Jun/'
#directory1='/Users/ahhmed/pytorch_unet/SPTmodels/23Jun_111055pm_150ep_23Jun/'

#chkPoint='24'
chkPoint='52'


NF=900#100#900 
nx=512
#N=400
N1=300
N2=50
nch=4
thres=0.05
nbasis=30
lam=0.001
st=0
bat_sz=100


#%%

trnFiles=os.listdir('./../Codes/TensorFlow_SToRM/Data/')
sz=len(trnFiles)
data=np.zeros((sz,nbasis,nx*nx)).astype(np.complex64)
rndm=random.sample(range(sz),sz)

for fl in range(sz):
    matfile = sio.loadmat('./../Codes/TensorFlow_SToRM/Data/'+trnFiles[rndm[fl]])
    #matfile = sio.loadmat('./../../../localscratch/Users/ahhmed/'+trnFiles[rndm[fl]])
    data1 = matfile['U1']
    data1=np.transpose(data1,(1,0))
    data1=data1[1:nbasis+1]
    data1=data1.astype(np.complex64)
    data1=data1/np.max(np.abs(data1))
    data[fl]=data1

U=np.stack((np.real(data),np.imag(data)),axis=1)
U=torch.tensor(U)
U=U.to(gpu)
U=torch.reshape(U,(sz,2*nbasis,nx,nx))


#%%
b_z=sz
G=UnetClass().to(gpu)
#z = torch.randn((sz,4,32,32),device=gpu, dtype=dtype
noise1=torch.normal(0,0.01,size=(U.shape[0],U.shape[1],U.shape[2],U.shape[3])).to(gpu)
z=U+noise1
z = Variable(z)
#optimizer=torch.optim.SGD([{'params':G.parameters(),'lr':5e-3,'momentum':0.9}])
loss_fn=torch.nn.MSELoss(reduction='sum')
optimizer=torch.optim.Adam([{'params':G.parameters(),'lr':1e-4}])
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=6, verbose=True, min_lr=5e-8)

#z1=z

for ep1 in range(4000):
    loss_F=torch.zeros((1))
    for bat in range(sz):
        u1=G(z[bat].unsqueeze(0))
        #U=U.permute(1,2,3,0)
        #u1=torch.reshape(u1,(2,nbasis,nx,nx))
        #u1=u1.permute(0,2,3,4,1)
        #u1=u1.permute(1,2,3,0)
        #u1=torch.reshape(u1,(nbasis,nx*nx*2))
        #loss=((u1-U)**2).sum()
        loss=loss_fn(u1[0],U[bat])
        #loss_F=loss_F+loss.item()
        if ep1%10==0: 
            print(ep1,loss.item())
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(loss.detach())
    #print(ep1,loss_F)

        
torch.save(G.state_dict(),'tempUfull_unet.pt')