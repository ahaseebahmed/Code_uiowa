#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 16:27:14 2021

@author: ahhmed
"""

import os,time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

import gen_netV as gnV

import supportingFun as sf
import cgFun as cf
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


from espirit.espirit import espirit, espirit_proj, ifft, fft

import scipy.io as sio

NF=900#100#900 
nx=512
nintl=10

trnFiles=os.listdir('/Shared/lss_jcb/abdul/prashant_cardiac_data/Data/d2/')
sz=len(trnFiles)
#data2=np.zeros((sz,1,n_select,N,N)).astype(np.complex64)

#rndm=random.sample(range(sz),sz)
#%%
#nuf_ob = KbNufft(im_size=(nx,nx)).to(dtype)
#nuf_ob=nuf_ob.to(gpu)
#
#adjnuf_ob = AdjKbNufft(im_size=(nx,nx)).to(dtype)
#adjnuf_ob=adjnuf_ob.to(gpu)

smapT=torch.ones((1,1,2,nx,nx)).cuda()

nuf_ob = MriSenseNufft(im_size=(nx,nx),smap=smapT).to(dtype)
nuf_ob=nuf_ob.to(gpu)

adjnuf_ob = AdjMriSenseNufft(im_size=(nx,nx), smap=smapT).to(dtype)
adjnuf_ob=adjnuf_ob.to(gpu)



d2=sio.loadmat('/Shared/lss_jcb/abdul/prashant_cardiac_data/Data/ktraj.mat')
ktraj=np.asarray(d2['ktraj'])
ktraj=ktraj.astype(np.complex64)
ktraj=np.transpose(ktraj,(2,1,0))/nx
ktraj=ktraj[0:NF]
ktraj=np.reshape(ktraj,(1,NF*nintl*nx))*2*np.pi
#dcf=np.abs(ktraj)

ktraj = np.stack((np.real(ktraj), np.imag(ktraj)), axis=1)
dcf = calculate_radial_dcomp_pytorch(nuf_ob, adjnuf_ob, torch.tensor(ktraj).cuda())
dcf=torch.tensor(dcf).unsqueeze(0).unsqueeze(0)
#dcf=dcf.repeat(nbasis,nch,2,1)
dcf=dcf.repeat(1,1,2,1)


ktraj=np.tile(ktraj,(nbasis,1,1))
ktrajT = torch.tensor(ktraj).to(dtype)


cc=torch.zeros((1,1,2,nx,nx)).cuda()
cc[0,0,:,255,255]=1
#cc[0,0,:,236,247]=1

tr=ktrajT[0].unsqueeze(0)
dd=nuf_ob(cc,tr.cuda())
ee1=adjnuf_ob(dd*dcf.cuda(),tr.cuda())
dcf=dcf/ee1[0,0,0,255,255]


dcf=torch.reshape(dcf,(2,NF,nintl,nx))
dcf[:,:,0]=0
dcf[:,:,1]=0
dcf=torch.reshape(dcf,(1,1,2,NF*nintl*nx))
#real_mat, imag_mat = precomp_sparse_mats(ktrajT, adjnuf_ob)
#interp_mats = {
#    'real_interp_mats': real_mat,
#    'imag_interp_mats': imag_mat
#}

#dcf=dcf+0.01
ktrajT_ch=ktrajT[0].unsqueeze(0)
dcf=dcf.repeat(nbasis,nch,1,1)
#df=torch.reshape(dcf[0,0,0],(NF,nintl*nx))
#df=df.unsqueeze(1).unsqueeze(1)
#df=df.repeat(1,3,2,1)
#df2=torch.reshape(df,(NF,3,2,nintl,nx))
#df2[:,:,:,0]=0
#df2[:,:,:,1]=0
#
#df2=torch.reshape(df2,(NF,3*2*nintl*nx))
        #%%
fl=10
str1=trnFiles[fl].split('_kd.mat')[0]
#dictn=sio.loadmat('/Shared/lss_jcb/abdul/prashant_cardiac_data/Data/'+trnFiles[rndm[fl]])
#d1=sio.loadmat('/Shared/lss_jcb/abdul/prashant_cardiac_data/Data/d2/'+str1+'_kd.mat')
#dictn=sio.loadmat('/Shared/lss_jcb/abdul/prashant_cardiac_data/Data/d1/'+str1+'.mat')
#dictn=sio.loadmat('/Shared/lss_jcb/abdul/prashant_cardiac_data/Data/meas_MID00185_FID142454_SAX_gNAV_gre11.mat')
#d1=sio.loadmat('/Shared/lss_jcb/abdul/prashant_cardiac_data/Data/kdata.mat')
d1=sio.loadmat('/Shared/lss_jcb/abdul/prashant_cardiac_data/Data/d2/'+str1+'_kd.mat')
dictn=sio.loadmat('/Shared/lss_jcb/abdul/prashant_cardiac_data/Data/d1/'+str1+'.mat')
d4=sio.loadmat('/Shared/lss_jcb/abdul/prashant_cardiac_data/Data/d4/'+str1+'_L900a.mat')
d3=sio.loadmat('/Shared/lss_jcb/abdul/prashant_cardiac_data/Data/d3/'+str1+'_mx.mat')


kdata=np.asarray(d1['kdata'])

kdata[:,:,0]=0
kdata[:,:,1]=0
kdata=np.transpose(kdata,(3,1,2,0))
kdata=kdata[0:NF,0:nch]
kdata=np.reshape(kdata,(NF,nch*nintl*nx))
kdata=kdata/d3['mx']
#kdata=kdata/np.max(np.abs(kdata))

L=np.asarray(d4['L']) 
L=L.astype(np.complex64)
L=L[0:NF,0:NF]
U,sb,V=np.linalg.svd(L)
#sb[NF-1]=0
V=V[TF-nbasis:TF,0+st:NF+st]

sb=np.diag(sb[NF-nbasis:NF,])
sb=torch.tensor(np.diag(sb))
V=V.astype(np.float32)

#V1=np.linalg.pinv(V@V.T)@V
#V2=V.T@V1
#V1=V.T@np.linalg.pinv(V@V.T)
#V1=V1.T
Vn=np.zeros((nbasis,NF,NF),dtype='float32')  
for i in range(nbasis):
    Vn[i,:,:]=np.diag(V[i,:])
Vn=np.reshape(Vn,(nbasis*NF,NF))
temp=Vn@kdata
temp=np.reshape(temp,(nbasis,NF,nch,nintl*nx))        
temp=np.transpose(temp,(0,2,1,3))
temp=np.reshape(temp,(nbasis,nch,NF*nintl*nx))
kdata=np.reshape(kdata,(NF,nch,nintl*nx))


#Tensors
#csmTrn[0].shape
kdata = np.stack((np.real(kdata), np.imag(kdata)),axis=2)
kda1 = np.stack((np.real(temp), np.imag(temp)),axis=2)

# convert to tensor, unsqueeze batch and coil dimension
kdataT = torch.tensor(kdata).to(dtype)
kdata_ch=kdataT.permute(1,2,0,3)
kdataT=torch.reshape(kdataT,(NF,nch,2,nx*nintl))
kdataT=kdataT.permute(1,2,0,3)
kdaT = torch.tensor(kda1).to(dtype)
kdaT=kdaT.to(gpu)       

#%      Coilimages ans csm estimation  

kdata_ch=torch.reshape(kdata_ch,(nch,2,NF*nintl*nx))
kdata_ch=kdata_ch.unsqueeze(0)
kdata_ch=kdata_ch.to(gpu)
ktrajT_ch=ktrajT_ch.to(gpu)
#dcfT=dcfT.to(gpu)        
#nuf_ob = KbNufft(im_size=im_size).to(dtype)
#nuf_ob=nuf_ob.to(gpu)

adjnuf_ob=adjnuf_ob.to(gpu)

coilimages=torch.zeros((1,nch,2,nx,nx))
#        A=lambda x: nuf_ob(x,ktrajT)
#        At=lambda x:adjnuf_ob(x,ktrajT)
#        AtA=lambda x:At(A(x))
for i in range(nch):
    coilimages[:,i]=adjnuf_ob(kdata_ch[:,i].unsqueeze(1),ktrajT_ch)
    #ini=torch.zeros_like(temp)
    #coilimages[:,i]=sf.pt_cg(AtA,temp,ini,50,1e-15)
X=coilimages.cpu().numpy()
x=X[:,:,0]+X[:,:,1]*1j
x=np.transpose(x,(2,3,0,1))
x_f = fft(x, (0, 1, 2))
csmTrn = espirit(x_f, 6, 24, 0.05, 0.9925)
csm=csmTrn[:,:,0,:,0]
csm=np.transpose(csm,(2,0,1))

smap = np.stack((np.real(csm), np.imag(csm)), axis=1)
smap=np.tile(smap,(nbasis,1,1,1,1))

nufft_ob = MriSenseNufft(im_size=im_size,smap=smapT).to(dtype)
nufft_ob=nufft_ob.to(gpu)

adjnufft_ob = AdjMriSenseNufft(im_size=im_size, smap=smapT).to(dtype)
adjnufft_ob=adjnufft_ob.to(gpu)

atb=torch.zeros((nbasis,im_size[0],im_size[1]))

# generating AHb results
#dcf=torch.tensor(dcf).unsqueeze(0).unsqueeze(0)
#dcf=dcf.repeat(1,1,2,1)
#dcf=dcf*2
atb=adjnufft_ob(kdaT*2*dcf.cuda(),ktrajT.cuda())