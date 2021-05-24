"""
This is the nufft 3D code in pytorch
@author: abdul haseeb
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import supportingFun as sf
import h5py as h5
from torchkbnufft.torchkbnufft import KbNufft
from torchkbnufft.torchkbnufft import AdjKbNufft
from torchkbnufft.torchkbnufft import MriSenseNufft
from torchkbnufft.torchkbnufft import AdjMriSenseNufft
from torchkbnufft.torchkbnufft.nufft.sparse_interp_mat import precomp_sparse_mats
import scipy.io as sio
from torch.utils.data import TensorDataset
from torch.utils.data import Dataset, DataLoader
from espirit.espirit import espirit, espirit_proj, ifft,fft



gpu=torch.device('cuda:0')

directory='/Users/ahhmed/pytorch_sense/3D_data/'
dtype = torch.float

NF=1
nx=200
nch=3
nkpts=128
nangles=206550
nkpts=383
nintl=44
nbasis=4

directory='/Users/ahhmed/pytorch_sense/Konnor_3D/'
dtype = torch.float

NF=300 
nx=300
nch=2
nkpts=383
nintl=44
nbasis=4
lam=0.0
#%%
#
#with h5.File(directory+'csm.h5', 'r') as f:  
#  # coil sensitivity maps
#  csm=np.asarray(f['/csm/re'])+np.asarray(f['/csm/im'])*1j
#  csm=csm.astype(np.complex64)
#  #csmTrn=np.transpose(csm,[0,2,1])
#  #csm=np.fft.fftshift(np.fft.fftshift(csm,1),2)
#  csmTrn=csm[0:nch,:,:]
#  ncoils=csmTrn.shape[0]
#  del csm
#  
#with h5.File(directory+'kdata.h5', 'r') as f:  
#  # reading the RHS: AHb
#  kdata=np.asarray(f['/kdata/re'])
#  kdata=kdata+np.asarray(f['/kdata/im'])*1j
#  kdata=kdata.astype(np.complex64)
#  kdata=np.transpose(kdata,(1,0))
#  kdata=np.squeeze(kdata[0:nch:])
#  kdata=np.reshape(kdata,(1,nch,nangles*nkpts))
#  #kdata2=np.transpose(kdata1,(0,1,3,2))
#  #kdata=np.reshape(kdata2,(1,nch,nangles*nkpts))
#
#with h5.File(directory+'dcf.h5', 'r') as f:  
#  # reading the RHS: AHb
#  dcf=np.asarray(f['/dcf/re'])
#  dcf=np.reshape(dcf,(1,nangles*nkpts))
#
#with h5.File(directory+'ktraj.h5', 'r') as f: 
#  ktraj=np.asarray(f['/ktraj/re'])
#  ktraj=np.squeeze(np.transpose(ktraj,[0,1]))*2*np.pi
#  ktraj=np.reshape(ktraj,(1,3,nangles*nkpts))



with h5.File(directory+'csm_300.h5', 'r') as f:  
  # coil sensitivity maps
  csm=np.asarray(f['/csm/re'])+np.asarray(f['/csm/im'])*1j
  csm=csm.astype(np.complex64)
  #csmTrn=np.transpose(csm,[0,2,1])
  #csm=np.fft.fftshift(np.fft.fftshift(csm,1),2)
  csmTrn=csm[0:nch,:,:]
  ncoils=csmTrn.shape[0]
  del csm
  
with h5.File(directory+'kdata.h5', 'r') as f:  
  # reading the RHS: AHb
  kdata=np.asarray(f['/kdata/re'])
  kdata=kdata+np.asarray(f['/kdata/im'])*1j
  kdata=kdata.astype(np.complex64)
  kdata=np.reshape(kdata,(7,NF,nintl*nkpts))
  kdata=np.squeeze(kdata[0:nch,0:NF,:])
  kdata=np.transpose(kdata,(1,0,2))
  #kdata=np.transpose(kdata,(1,0,2))  
#kdata=np.reshape(kdata,(nch,NF*6*2336)) 

#with h5.File(directory+'y.h5', 'r') as f:  
#  # reading the RHS: AHb
#  y=np.asarray(f['/y/re'])
#  y=y+np.asarray(f['/y/im'])*1j
#  y=y.astype(np.complex64)
#  y=np.squeeze(np.transpose(y[:,:,:],[0,1,2]))  
#kdata=np.reshape(kdata,(3,NF*6*2336)) 

with h5.File(directory+'dcf.h5', 'r') as f:  
  # reading the RHS: AHb
  dcf=np.asarray(f['/dcf/re'])
  dcf=np.reshape(dcf,(1,NF*nintl*nkpts))
  dcf=np.tile(dcf,(2,1))

with h5.File(directory+'ktraj.h5', 'r') as f: 
  ktraj=np.asarray(f['/ktraj/re'])
  #ktraj=ktraj+np.asarray(f['/csm/im'])*1j
  #ktraj=ktraj.astype(np.complex64)
  ktraj=np.squeeze(np.transpose(ktraj,[1,0]))*2*np.pi
ktraj=np.reshape(ktraj,(1,3,NF*nintl*nkpts))
#ktraj=np.transpose(ktraj,(0,1))  
#ktraj=np.tile(ktraj,(nbasis,1,1,))

with h5.File(directory+'L.h5', 'r') as f:  
  # reading the RHS: AHb
  L=np.asarray(f['/L'])
  U,sbasis,V=np.linalg.svd(L)
  V=V[NF-nbasis:NF,:]
  sbasis=np.diag(sbasis[NF-nbasis:NF,])*lam
Vn=np.zeros((nbasis,NF,NF))  
for i in range(nbasis):
    Vn[i,:,:]=np.diag(V[i,:])
#%%
im_size = csmTrn[0].shape

kdata = np.stack((np.real(kdata), np.imag(kdata)),axis=2)
kdataT = torch.tensor(kdata).to(dtype)

dcfT = torch.tensor(dcf).to(dtype)

smap = np.stack((np.real(csmTrn), np.imag(csmTrn)), axis=1)
smapT = torch.tensor(smap).to(dtype)

ktrajT = torch.tensor(ktraj).to(dtype)

#%% take them to gpu
kdataT=kdataT.to(gpu)
smapT=smapT.to(gpu)
smapT=smapT.unsqueeze(0)
ktrajT=ktrajT.to(gpu)
dcfT=dcfT.to(gpu)
#%% generate atb
kdataT=kdataT.permute(1,2,0,3)
kdataT=torch.reshape(kdataT,(nch,2,NF*nintl*nkpts))
kdataT=kdataT.unsqueeze(0)

nuf_ob = KbNufft(im_size=im_size).to(dtype)
nuf_ob=nuf_ob.to(gpu)
adjnuf_ob = AdjKbNufft(im_size=im_size).to(dtype)
adjnuf_ob=adjnuf_ob.to(gpu)
#real_mat, imag_mat = precomp_sparse_mats(ktrajT, adjnuf_ob)
#interp_mats = {'real_interp_mats': real_mat, 'imag_interp_mats': imag_mat}

coilimages=torch.zeros((1,nch,2,nx,nx,nx))
A=lambda x: nuf_ob(x,ktrajT)
At=lambda x:adjnuf_ob(x,ktrajT)
AtA=lambda x:At(A(x))
for i in range(nch):
    temp=adjnuf_ob(kdataT[:,i].unsqueeze(1)*dcfT,ktrajT[0].unsqueeze(0))
    ini=torch.zeros_like(temp)
    coilimages[:,i]=sf.pt_cg(AtA,temp,ini,2,1e-15)
X=coilimages.cpu().numpy()
x=X[:,:,0]+X[:,:,1]*1j
x=np.transpose(x[0],(1,2,3,0))
x_f = fft(x, (0, 1, 2))
csmTrn = espirit(x_f, 6, 32, 0.01, 0.95)
csm=csmTrn[:,:,:,:,0]
csm=np.transpose(csm,(2,0,1))
smap = np.stack((np.real(csm), np.imag(csm)), axis=1)
smap=np.tile(smap,(nbasis,1,1,1,1))
smapT = torch.tensor(smap).to(dtype)
smapT=smapT.to(gpu)
#%%
nufft_ob = MriSenseNufft(im_size=im_size, smap=smapT).to(dtype)
nufft_ob=nufft_ob.to(gpu)
adjnufft_ob = AdjMriSenseNufft(im_size=im_size, smap=smapT).to(dtype)
adjnufft_ob=adjnufft_ob.to(gpu)
#nufft_ob = MriSenseNufft(smap=smapT,im_size=im_size).to(dtype)
#adjnufft_ob = AdjMriSenseNufft(smap=smapT,im_size=im_size ).to(dtype)

#At=lambda x: adjnufft_ob(x*dcfT,ktrajT,interp_mats)
#A=lambda x: nufft_ob(x,ktrajT,interp_mats)

At=lambda x: adjnufft_ob(x*dcfT,ktrajT)
A=lambda x: nufft_ob(x,ktrajT)
#kdataT1=A(yT)
atbT=At(kdataT)
#del kdataT

#%% Loading network parameters
#unet=SmallModel()
#modelDir=directory1+'wts-'+str(chkPoint)+'.pt'
#unet.load_state_dict(torch.load(modelDir))
#unet.eval()
#unet.to(gpu)
#
#def dn_net(xx):
#    y=[]
#    trnDs=TensorDataset(torch.tensor(xx),torch.tensor(xx))
#    ldr=DataLoader(trnDs,batch_size=1,shuffle=False,num_workers=8)
#    with torch.no_grad():
#        for data in (ldr):
#            slcR=unet(data[0].to(gpu,dtype))
#            y.append(slcR)
#    Y=torch.cat(y)
#    xx=Y.detach().cpu().numpy()
#    return xx
##%% run cg-sense
#recT=atbT
#lam=0.01
#cgIter=10
#cgTol=1e-15
#
#B=lambda x: At(A(x))+lam*x
#sf.tic()
#for i in range(2):
#    atbT1=atbT+lam*dn_net(recT)
#    recT=sf.pt_cg(B,atbT,cgIter,cgTol)
#sf.toc()
#
#rec = np.squeeze(atbT.cpu().numpy())
#rec = rec[0] + 1j*rec[1]
##%%
#xx=[np.abs(rec), np.angle(rec)]
#for i in range(50,150):
#    plt.imshow((np.squeeze(np.imag(rec[:,:,i]))),cmap='gray')       
#    plt.show() 
#    plt.pause(0.01)