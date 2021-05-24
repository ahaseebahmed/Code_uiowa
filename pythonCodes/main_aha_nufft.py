"""
Created on Tue Apr 14 11:36:39 2020
This is the sense code in pytorch
@author: haggarwal
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
  
with h5.File('kdata.h5', 'r') as f:  
  # reading the RHS: AHb
  kdata=np.asarray(f['/kdata/re'])
  kdata=kdata+np.asarray(f['/kdata/im'])*1j
  kdata=kdata.astype(np.complex64)
  kdata=np.reshape(kdata,(nch,NF,1,nintl*nkpts))
  kdata=np.reshape(kdata,(nch,NF,1*nintl*nkpts))
  kdata=np.squeeze(kdata[0:nch,0:NF,:])
  kdata=np.transpose(kdata,(1,0,2))  
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
  dcf=np.reshape(dcf,(NF,1*nintl*nkpts))
  dcf=np.tile(dcf,(3,2,1,1))
  dcf=np.transpose(dcf,(2,0,1,3))

with h5.File('ktraj.h5', 'r') as f: 
  ktraj=np.asarray(f['/ktraj/re'])
  ktraj=ktraj+np.asarray(f['/csm/im'])*1j
  ktraj=ktraj.astype(np.complex64)
  ktraj=np.reshape(ktraj,(NF,1,nintl*nkpts))
  ktraj=np.reshape(ktraj,(NF,1*nintl*nkpts))
  ktraj=np.squeeze(np.transpose(ktraj[0:NF,:],[0,1]))*2*np.pi
#ktraj=np.reshape(ktraj,(NF,nintl*nkpts))  


with h5.File(directory+'L.h5', 'r') as f:  
  # reading the RHS: AHb
  L=np.asarray(f['/L'])
  U,sbasis,V=np.linalg.svd(L)
  V=V[NF-nbasis:NF,:]
  sbasis=sbasis[NF-nbasis:NF,]
#%%
im_size = csmTrn[0].shape

kdata = np.stack((np.real(kdata), np.imag(kdata)),axis=2)
# convert to tensor, unsqueeze batch and coil dimension
# output size: (1, 1, 2, ny, nx)
kdataT = torch.tensor(kdata).to(dtype)

dcfT = torch.tensor(dcf).to(dtype)
LT = torch.tensor(L).to(dtype)


smap = np.stack((np.real(csmTrn), np.imag(csmTrn)), axis=1)
smap=np.tile(smap,(NF,1,1,1,1))
smapT = torch.tensor(smap).to(dtype)
# convert to tensor, unsqueeze batch dimension
# output size: (1, 2, klength)
ktraj = np.stack((np.real(ktraj), np.imag(ktraj)), axis=1)
ktrajT = torch.tensor(ktraj).to(dtype)

#y = np.stack((np.real(y), np.imag(y)), axis=0)
#yT = torch.tensor(y).to(dtype).unsqueeze(0).unsqueeze(0)

#%% take them to gpu
kdataT=kdataT.to(gpu)
smapT=smapT.to(gpu)
ktrajT=ktrajT.to(gpu)
dcfT=dcfT.to(gpu)
LT=LT.to(gpu)
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

At=lambda x: adjnufft_ob(x*dcfT,ktrajT,interp_mats)
A=lambda x: nufft_ob(x,ktrajT,interp_mats)
#At=lambda x: adjnuf_ob(x,ktrajT)
#A=lambda x: nuf_ob(x,ktrajT)
#kdataT1=A(yT)
atbT=At(kdataT)
#del kdataT
#xx=sf.complex_multiplication(atbT,LT)
#xx=xx.to(gpu)
#%% run cg-sense
B=lambda x: (At(A(x))+lam*((sf.complex_multiplication(x,L,NF)).to(gpu)))
sf.tic()
recT=sf.pt_cg(B,atbT,cgIter,cgTol)
sf.toc()

rec = np.squeeze(recT.cpu().numpy())
rec = rec[:,0,] + 1j*rec[:,1,]
#%%
for i in range(100):
    plt.imshow((np.squeeze(np.abs(rec[i,:,:]))),cmap='gray')       
    plt.show() 
    plt.pause(0.1)
    
    

fn=lambda x:  sf.normalize01(np.abs(sf.r2c(x.cpu().numpy())))
normOrg=fn(orgT)
normAtb=fn(atbT)
normRec=fn(recT)

psnrAtb=sf.myPSNR(normOrg,normAtb)
#ssimAtb=sf.mySSIM(normOrg,normAtb)

psnrRec=sf.myPSNR(normOrg,normRec)
#ssimRec=sf.mySSIM(normOrg,normRec)
print ('  ' + 'Noisy ' + 'Rec')
print ('  {0:.2f} {1:.2f}'.format(psnrAtb.mean(),psnrRec.mean()))
#print ('  {0:.2f} {1:.2f}'.format(ssimAtb.mean(),ssimRec.mean()))
#%% view the results

fig,ax=plt.subplots(1,3)
ax[0].imshow(normOrg)
ax[0].set_title('original')
ax[0].axis('off')
ax[1].imshow(normAtb)
ax[1].set_title('Atb ')
ax[1].axis('off')
ax[2].imshow(normRec)
ax[2].set_title('Recon ')
ax[2].axis('off')
plt.show()

#%%

def Atb_UV(FT,kspT,ktraj,V,im_size,nB):
    atb=torch.zeros((nB,im_size))
    for i in range(nB):
        temp=torch.matmul(torch.diag(V[i,:]),kspT[:,:,:,:])
        atb[i,]=FT(temp,ktraj)
    
    return atb


def AtA_UV(FT,FTt,x,V,im_size,nsampfrm,nB,NF):
    
    y=torch.zeros(nB,im_size)
    vksp=torch.zeros(NF,nsampfrm)
    for i in range(nB):
        temp=FT(x[i,])
        vksp=vksp +torch.matmul(torch.diag(V[i,:]),temp)
        
    for j in range(nB):
        temp=torch.matmul(torch.diag(V[j,:]),vksp)
        y[j,]=y[j,]+FTt(temp)