#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 10:49:17 2021

@author: ahhmed
"""

import os,time
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import scipy.io as sio
import sys

sys.path.insert(0, "/Users/ahhmed/pytorch_unet")

from torch.optim.lr_scheduler import ReduceLROnPlateau

import scipy.io as sio
from scipy.linalg import fractional_matrix_power

from dn_modelV2_IR import SmallModel

from scipy import ndimage
gpu=torch.device('cuda:0')
cpu=torch.device('cpu')
directory='/Users/ahhmed/Codes/TensorFlow_SToRM/Data1/FID142454v/d900/'
directory2='/Users/ahhmed/Codes/TensorFlow_SToRM/Data1/FID142454v/'

dtype = torch.float
directory1='/Users/ahhmed/pytorch_unet/SPTmodels/19Jun_101938pm_50ep_20Jun/'
#directory1='/Users/ahhmed/pytorch_unet/SPTmodels/21Jun_110930pm_100ep_21Jun/'
#directory1='/Users/ahhmed/pytorch_unet/SPTmodels/23Jun_111055pm_150ep_23Jun/'

NF=1000#100#900 
nx=512
nintl=10
#N=400
N1=300
N2=50
nch=3
thres=0.05
nbasis=32
lam=0.01
st=0
batch_sz=100
TF=1000
thres=0.4
im_size = [nx,nx]
nf1=300
#%%
dictn=sio.loadmat('Spiral_cine_3T_1leave_FB_062_IR_FA15_TBW5_6_fixedSpiral_post_full_V.mat')

L=np.asarray(dictn['L']) 
L=L.astype(np.complex64)
L=L[0:NF,0:NF]
U,sb,V=np.linalg.svd(L)
#sb[NF-1]=0
V=V[TF-nbasis:TF,0+st:NF+st]

sb=np.diag(sb[NF-nbasis:NF,])*lam
sb=torch.tensor(np.diag(sb))
V=V.astype(np.float32)


def KLloss(zvec1):
    loss = 0
    #Nsamples = 2
    #for j in range(Nsamples):
    mn = torch.mean(zvec1,1) #nf x latentvecotsrs
    meansub = zvec1 - mn.unsqueeze(1)
    Sigma = meansub@meansub.T/zvec1.shape[1]
    tr = torch.trace(Sigma)
    if(tr> 0.01):
        loss = loss+ 0.5*(mn@mn.T + tr - zvec1.shape[0] - torch.logdet(Sigma))
    return(loss)   
#%%

z1=torch.normal(0,0.1,(1,1,8,NF-4)).cuda()
#z1=0.01*torch.ones((1,1,8,NF)).to(gpu)
##z1[0,0,1]=-100
#
z1[0,0,:,:]=100*torch.tensor(V[22:30,4:NF])
#z1[0,0,0,:]=torch.tensor(V[27,0:NF])

#z1[0,0,1,:]=0.1*torch.sin(torch.tensor(0.7*np.asarray(range(300))))
#z1[0,0,0,:]=0.1*torch.sin(torch.tensor(0.1*np.asarray(range(300))))

#vv=V[30:32]
#z1=torch.reshape(torch.tensor(V[29:31,0:nf1]),(1,1,2,nf1)).to(gpu)
z1 = Variable(z1,requires_grad=True)
V1=torch.tensor(V[:,4:NF]).cuda()#+torch.normal(0,0.01,(V.shape[0],V.shape[1])).cuda()
#V1=torch.reshape(V1,(1,1,nbasis,nf1))

GV=SmallModel(16).to(gpu)
#GV=generatorV(2).to(gpu)
optimizer=torch.optim.Adam([{'params':GV.parameters(),'lr':1e-4},{'params':z1,'lr':1e-3}])
#optimizer=torch.optim.Adam([{'params':GV.parameters(),'lr':1e-4}])
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=6, verbose=True, min_lr=1e-7)
pp=np.array([])
for ep1 in range(50000):#25000
    for bat in range(1):
        
        #z1=torch.reshape(V1,(1,1,nbasis,nf1))
        l1_reg=0.
        for param in GV.parameters():
            l1_reg += param.abs().sum()  
        v1=GV(z1)
        #v1=v1.permute(1,0)
        loss=abs(v1[0,0]-V1).pow(2).sum()+10.0*KLloss(z1[0,0])#+1.0*Smoothness(z1[0,0,:,:]) 
        if ep1%10==0:
            print(ep1,loss.item())
        
#        if ep1%5000==0:
#            plt.figure(ep1)
#            dd=(v1[0,0,:,20].unsqueeze(1).unsqueeze(1).unsqueeze(1)*recT1.squeeze(1)).sum(dim=0).detach().cpu().numpy()
#            plt.imshow(np.abs((dd[0,150:350,150:300]+dd[1,150:350,150:300]*1j)),cmap='gray')
#            plt.show()
#            plt.pause(1)
            
#        if ep1%501==0:
#            plt.figure(ep1)
#            zz=z1.detach().cpu().numpy()
#            plt.plot(zz[0,0,0])
#            plt.plot(zz[0,0,1])
#            plt.show()
#            plt.pause(1)
    
#        dd=(v1[0,0,:,20:30].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).detach().cpu()*recT1[:,:,:,150:350,150:300].detach().cpu()).sum(dim=0).detach().cpu()
#        org1=(v2[0,0,:,20:30].unsqueeze(-1).unsqueeze(-1).detach().cpu()*recT[:,:,:,150:350,150:300].detach().cpu()).sum(dim=0)
#        psnr=torch.mean(myPSNR(org1,dd))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(ep1,loss.item())
        #pp=np.append(pp)
        scheduler.step(loss.detach())
        
#%%
#z1=torch.normal(0,0.01,(1,1,2,nf1)).cuda()
Vx=np.squeeze(v1.detach().cpu().numpy())
Vz=V[:,4:NF]#V.detach().cpu().numpy()

for i in range(6):
    plt.figure(i)
    plt.plot(Vz[i])
    plt.pause(0.1)
    plt.plot(Vx[i])
    #plt.pause()       
    plt.show() 