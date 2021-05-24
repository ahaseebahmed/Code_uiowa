"""
Created on Tue Apr 14 11:36:39 2020
This is the cart_UV code in pytorch
@author: abdul haseeb
"""
import os,time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import gen_net as gn
import gen_net3 as gn2
import gen_netV as gnV
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import supportingFun as sf
import supporting_storm_fun as ss
from torch.autograd import Variable
import h5py as h5
import readData as rd
import scipy.io as sio
import sys
from torch.utils.data import TensorDataset
import random

sys.path.insert(0, "/Users/ahhmed/pytorch_unet")

from torchkbnufft.torchkbnufft import KbNufft
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torchkbnufft.torchkbnufft import KbNufft
from torchkbnufft.torchkbnufft import AdjKbNufft, ToepNufft
from torchkbnufft.torchkbnufft import MriSenseNufft
from torchkbnufft.torchkbnufft import AdjMriSenseNufft, ToepSenseNufft
from torchkbnufft.torchkbnufft.nufft.sparse_interp_mat import precomp_sparse_mats
from torchkbnufft.torchkbnufft.nufft.toep_functions import calc_toep_kernel
from torchkbnufft.torchkbnufft.mri.dcomp_calc import calculate_radial_dcomp_pytorch
#from ..functional.kbnufft import AdjKbNufftFunction, KbNufftFunction


#from ..math import complex_mult


from espirit.espirit import espirit, espirit_proj, ifft, fft

import scipy.io as sio
from scipy.linalg import fractional_matrix_power

from unetmodel2 import UnetClass
from dn_modelV2 import SmallModel
from dn_modelU2 import SmallModel1


from scipy import ndimage
import gen_net1 as gn1
gpu=torch.device('cuda:0')
cpu=torch.device('cpu')
directory='/Users/ahhmed/Codes/TensorFlow_SToRM/Data1/FID142454v/d900/'
directory2='/Users/ahhmed/Codes/TensorFlow_SToRM/Data1/FID142454v/'

dtype = torch.float
directory1='/Users/ahhmed/pytorch_unet/SPTmodels/19Jun_101938pm_50ep_20Jun/'
#directory1='/Users/ahhmed/pytorch_unet/SPTmodels/21Jun_110930pm_100ep_21Jun/'
#directory1='/Users/ahhmed/pytorch_unet/SPTmodels/23Jun_111055pm_150ep_23Jun/'

#chkPoint='24'
chkPoint='52'


NF=900#100#900 
nx=512
nintl=10
#N=400
N1=300
N2=50
nch=3
thres=0.05
nbasis=30
lam=0.01
st=0
batch_sz=100
TF=900
im_size = [nx,nx]
noise_level=[0.1,0.05,0.09]

#%%

Inp=torch.load('UB.pt')

#mx=torch.max((atb**2).sum(dim=-3).sqrt())

G=SmallModel1().to(gpu)
#G.load_state_dict(torch.load('wtsUB-50.pt'))
#GV=SmallModel().to(gpu)

optimizer=torch.optim.AdamW([{'params':G.parameters(),'lr':1e-3}])
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=6, verbose=True, min_lr=1e-5)
#%%
trnInp=torch.zeros((len(noise_level),Inp.shape[0],Inp.shape[1],Inp.shape[2],Inp.shape[3]))
for n1 in range(len(noise_level)):
    #noise1=np.random.normal(0,noise_level[n1],(Inp.shape[0],Inp.shape[1],Inp.shape[2],Inp.shape[3]))
    #noise1=torch.normal(0,noise_level[nn],size=(inp.shape[0],inp.shape[1],inp.shape[2],inp.shape[3],inp.shape[4]))
    trnInp[n1]=Inp + torch.normal(0,noise_level[n1],(Inp.shape[0],Inp.shape[1],Inp.shape[2],Inp.shape[3]))
Inp=Inp.unsqueeze(0)
Inp=Inp.repeat(len(noise_level),1,1,1,1)
trnInp=torch.reshape(trnInp,(len(noise_level)*Inp.shape[1],Inp.shape[2],Inp.shape[3],Inp.shape[4]))
Inp=torch.reshape(Inp,(trnInp.shape[0],trnInp.shape[1],trnInp.shape[2],trnInp.shape[3]))
trnDs=TensorDataset(trnInp,Inp)
#trnDs=rd.DataGen(trnInp,trnOrg)
loader=DataLoader(trnDs,batch_size=1,shuffle=True,num_workers=8,pin_memory=True)

for ep in range(500):
    epLoss = 0.0
    epStart=time.time()

    for i, (inp,org) in enumerate(loader):
        org,inp = org.to(gpu,dtype=torch.float), inp.to(gpu,dtype=torch.float)
        pred = G(inp)
        loss=abs(pred-org).pow(2).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(loss.detach())
        epLoss += loss.item()
    epEnd=time.time()
    epTime=(epEnd - epStart) / 60
    print('Epoch:%d, loss: %.3f, time: %.2f min' % (ep + 1, epLoss,epTime))
#writer.add_scalar('training_loss', epLoss,global_step=ep+1)

wtsFname='wtsUB-'+str(ep+1)+'.pt'
torch.save(G.state_dict(), wtsFname)
#writer.close()




        #kdataT=kdataT/mx
        
        #dcomp = calculate_radial_dcomp_pytorch(nufft_ob, adjnufft_ob, xx.cuda()).unsqueeze(0).unsqueeze(0)
#        toep_ob = ToepSenseNufft(smap=smapT)
#        dcomp_kern = calc_toep_kernel(adjnufft_ob, ktrajT.cuda(), weights=dcf.cuda())
#        image_sharp_toep = toep_ob(atb, dcomp_kern)
        #z=atb+torch.normal(0,0.01,(atb.shape[0],atb.shape[1],atb.shape[2],atb.shape[3])).cuda()
#        #z=torch.reshape(z,(1,2*nbasis,nx,nx))
#        for it in range(500):
#            z=atb+torch.normal(0,0.005,(atb.shape[0],atb.shape[1],atb.shape[2],atb.shape[3])).cuda()
#            z=torch.reshape(z,(1,2*nbasis,nx,nx))
#            u1=G(z.cuda())
#            u1=torch.reshape(u1,(nbasis,2,nx,nx))
#            loss=abs(u1-atb).pow(2).sum()
#            print(it,loss.item())
#            #torch.cuda.empty_cache()
#            optimizer.zero_grad()
#            loss.backward()
#            optimizer.step()
#            scheduler.step(loss.detach())
        
        
        #%%


#    if ep1%2==0:
#        wtsFname1='wts-10U'+str(ep1+1)+'.pt'
#        torch.save(G.state_dict(),wtsFname1) 
#        wtsFname2='wts-10V'+str(ep1+1)+'.pt'

