"""
Created on Tue Apr 14 11:36:39 2020
This is the sense code in pytorch
@author: haggarwal
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import h5py as h5
from torchkbnufft.torchkbnufft import KbNufft
from torchkbnufft.torchkbnufft import AdjKbNufft
from torchkbnufft.torchkbnufft import MriSenseNufft
from torchkbnufft.torchkbnufft import AdjMriSenseNufft
from torchkbnufft.torchkbnufft.nufft.sparse_interp_mat import precomp_sparse_mats


gpu=torch.device('cuda:0')

directory='/Users/ahhmed/pytorch_sense/'
dtype = torch.float

NF=200 
nx=340
nch=3
nkpts=2336
nintl=6
nbasis=30
#%%

with h5.File(directory+'csm_se17.h5', 'r') as f:  
  # coil sensitivity maps
  csm=np.asarray(f['/csm/re'])+np.asarray(f['/csm/im'])*1j
  csm=csm.astype(np.complex64)
  #csm=np.transpose(csm,(0,2,1))
  #csmTrn=np.transpose(csm,[0,2,1])
  #csm=np.fft.fftshift(np.fft.fftshift(csm,1),2)
  csmTrn=csm[0:nch,:,:]
  ncoils=csmTrn.shape[0]
  del csm
  
with h5.File(directory+'kdata_se17.h5', 'r') as f:  
  # reading the RHS: AHb
  kdata=np.asarray(f['/kdata/re'])
  kdata=kdata+np.asarray(f['/kdata/im'])*1j
  kdata=kdata.astype(np.complex64)
  kdata=np.reshape(kdata,(nch,NF,1,nintl*nkpts))
  kdata=np.reshape(kdata,(nch,NF,1*nintl*nkpts))
  kdata=np.squeeze(kdata[0:nch,0:NF,:])
  kdata=np.transpose(kdata,(1,0,2))  
#kdata=np.reshape(kdata,(nch,NF*6*2336)) 
with h5.File(directory+'dcf.h5', 'r') as f:  
  # reading the RHS: AHb
  dcf=np.asarray(f['/dcf/re'])
  dcf=np.reshape(dcf,(NF,1*nintl*nkpts))
  dcf=np.tile(dcf,(3,2,1,1))
  dcf=np.transpose(dcf,(2,0,1,3))

with h5.File(directory+'ktraj_se17.h5', 'r') as f: 
  ktraj=np.asarray(f['/csm/re'])
  ktraj=ktraj+np.asarray(f['/csm/im'])*1j
  ktraj=ktraj.astype(np.complex64)
  ktraj=np.reshape(ktraj,(NF,1,nintl*nkpts))
  ktraj=np.reshape(ktraj,(NF,1*nintl*nkpts))
  ktraj=np.squeeze(np.transpose(ktraj[0:NF,:],[0,1]))*2*np.pi
#ktraj=np.reshape(ktraj,(NF,nintl*nkpts))  

#%%
im_size = csmTrn[0].shape

kdata = np.stack((np.real(kdata), np.imag(kdata)),axis=2)
# convert to tensor, unsqueeze batch and coil dimension
# output size: (1, 1, 2, ny, nx)
kdataT = torch.tensor(kdata).to(dtype)

dcfT = torch.tensor(dcf).to(dtype)


smap = np.stack((np.real(csmTrn), np.imag(csmTrn)), axis=1)
smap=np.tile(smap,(50,1,1,1,1))
smapT = torch.tensor(smap).to(dtype)
# convert to tensor, unsqueeze batch dimension
# output size: (1, 2, klength)
ktraj = np.stack((np.real(ktraj), np.imag(ktraj)), axis=1)
ktrajT = torch.tensor(ktraj).to(dtype)

#%% take them to gpu
kdataT=kdataT.to(gpu)
smapT=smapT.to(gpu)

ktrajT=ktrajT.to(gpu)
dcfT=dcfT.to(gpu)
#%% generate atb
sigma=0.0
lam=1e3
cgIter=10
cgTol=1e-15

nuf_ob = KbNufft(im_size=im_size).to(dtype)
nuf_ob=nuf_ob.to(gpu)
adjnuf_ob = AdjKbNufft(im_size=im_size).to(dtype)
adjnuf_ob=adjnuf_ob.to(gpu)
real_mat, imag_mat = precomp_sparse_mats(ktrajT, adjnuf_ob)
interp_mats = {'real_interp_mats': real_mat, 'imag_interp_mats': imag_mat}

nufft_ob = MriSenseNufft(im_size=im_size, smap=smapT).to(dtype)
nufft_ob=nufft_ob.to(gpu)
adjnufft_ob = AdjMriSenseNufft(im_size=im_size, smap=smapT).to(dtype)
adjnufft_ob=adjnufft_ob.to(gpu)
#nufft_ob = MriSenseNufft(smap=smapT,im_size=im_size).to(dtype)
#adjnufft_ob = AdjMriSenseNufft(smap=smapT,im_size=im_size ).to(dtype)

#At=lambda x: adjnufft_ob(x*dcfT,ktrajT,interp_mats)
#A=lambda x: nufft_ob(x,ktrajT,interp_mats)
At=lambda x: adjnufft_ob(x,ktrajT)
A=lambda x: nufft_ob(x,ktrajT)
#kdataT1=A(yT)
kdataT1=torch.reshape(kdataT,(4,50,3,2,14016))
#kdataT1=kdataT1.permute(1,0,2,3,4)
ktrajT1=torch.reshape(ktrajT,(4,50,2,14016))
#ktrajT1=ktrajT1.permute(1,0,2,3)
dcfT1=torch.reshape(dcfT,(4,50,3,2,14016))
#dcfT1=dcfT1.permute(1,0,2,3,4)

def At(kdataT1,ktrajT1):
    y=torch.zeros((4,50,1,2,nx,nx)).to(gpu,dtype)
    for i in range(4):
        y[i]=adjnufft_ob(kdataT1[i],ktrajT1[i])
    return y

def A(xx,ktrajT1):
    yy=torch.zeros((4,50,3,2,14016)).to(gpu,dtype)
    for i in range(4):
        yy[i]=nufft_ob(xx[i],ktrajT1[i])
    return yy


atbT=At(kdataT1*dcfT1,ktrajT1) #inverse NUFFT^ transform
kdataT_new=A(atbT,ktrajT1)#NUFFT transform
