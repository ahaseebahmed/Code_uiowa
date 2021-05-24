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
#from spatial_model6 import SmallModel
from unetmodel import UnetClass
from dn_modelV import SmallModel



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
data=np.zeros((sz,nbasis,NF))
rndm=random.sample(range(sz),sz)

for fl in range(sz):
    matfile = sio.loadmat('./../Codes/TensorFlow_SToRM/Data/'+trnFiles[rndm[fl]])
    #matfile = sio.loadmat('./../../../localscratch/Users/ahhmed/'+trnFiles[rndm[fl]])
    data1 = matfile['D']
    data1=np.transpose(data1,(1,0))
    data1=data1[1:nbasis+1]
    #data1=data1.astype(np.complex64)
    #data1=data1/np.max(np.abs(data1))
    data[fl]=data1

#V=np.stack((np.real(data),np.imag(data)),axis=1)
V=torch.tensor(data)
V=V.to(gpu)
V=torch.reshape(V,(sz,1,nbasis,NF))
V=V.to(torch.float32)

#%%
noise_level=[0.01,0.05,0.001,0.0001]
b_z=sz
GV=SmallModel().to(gpu)
#optimizer=torch.optim.SGD([{'params':G.parameters(),'lr':5e-3,'momentum':0.9}])
loss_fn=torch.nn.MSELoss(reduction='sum')
optimizer=torch.optim.Adam([{'params':GV.parameters(),'lr':1e-4}])
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=6, verbose=True, min_lr=5e-8)

#z = torch.randn((sz,4,32,32),device=gpu, dtype=dtype
#noise1=torch.normal(0,0.01,size=(V.shape[0],V.shape[1],V.shape[2],V.shape[3])).to(gpu)


for ep1 in range(2000):
    loss_F=torch.zeros((1))
    for n1 in range(len(noise_level)):
        noise1=torch.normal(0,noise_level[n1],size=(V.shape[0],V.shape[1],V.shape[2],V.shape[3])).to(gpu)
        z=V+noise1
        z = Variable(z)
        for bat in range(sz):
            v1=GV(z[bat].unsqueeze(0))
            #U=U.permute(1,2,3,0)
            #u1=torch.reshape(u1,(2,nbasis,nx,nx))
            #u1=u1.permute(0,2,3,4,1)
            #u1=u1.permute(1,2,3,0)
            #u1=torch.reshape(u1,(nbasis,nx*nx*2))
            #loss=((u1-U)**2).sum()
            loss=loss_fn(v1[0],V[bat])
            #loss_F=loss_F+loss.item()
            if ep1%10==0: 
                print(ep1,loss.item())
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step(loss.detach())
    #print(ep1,loss_F)

        
torch.save(GV.state_dict(),'tempVfull_dn.pt')