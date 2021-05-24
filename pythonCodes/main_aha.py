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

gpu=torch.device('cuda:0')

directory='/Users/ahhmed/Codes/TensorFlow_SToRM/Data1/FID142454v/'

NF=2 
nx=512
#N=400
N1=300
N2=50
nch=5
thres=0.05

#%%

with h5.File(directory+'csm.h5', 'r') as f:  
  # coil sensitivity maps
  csm=np.asarray(f['/csm/re'])+np.asarray(f['/csm/im'])*1j
  csm=csm.astype(np.complex64)
  csmTrn=np.transpose(csm,[0,2,1])
  #csm=np.fft.fftshift(np.fft.fftshift(csm,1),2)
  csmTrn=csmTrn[0:nch,:,:]
  ncoils=csmTrn.shape[0]
  del csm
  
with h5.File(directory+'rhs.h5', 'r') as f:  
  # reading the RHS: AHb
  atb=np.asarray(f['/rhs/re'])
  atb=atb+np.asarray(f['/rhs/im'])*1j
  atb=atb.astype(np.complex64)
  atb=np.squeeze(np.transpose(atb[0:NF,:,:],[0,2,1]))  

with h5.File(directory+'S_mask.h5', 'r') as f: 
  maskTst=np.asarray(f['S'])
  maskTst=np.squeeze(maskTst[0:NF,:,:])



#org=np.load('data.npz')['org']
#csm=np.load('data.npz')['csm']
#mask=np.load('data.npz')['mask']

csm=csmTrn
mask=maskTst

#%% make them pytorch tensors
orgT=torch.tensor(sf.c2r(atb))
csmT=torch.tensor(sf.c2r(csm))
maskT=torch.tensor(np.tile(mask[...,np.newaxis],(1,1,2)))

#%% take them to gpu
orgT=orgT.to(gpu)
csmT=csmT.to(gpu)
maskT=maskT.to(gpu)

#%% make A and At operators
def pt_A(orgT,csmT,maskT):
    tmp=sf.pt_cpx_multipy(orgT,csmT)
    tmp=sf.pt_fft2c(tmp)
    bT=maskT*tmp
    return bT


def pt_At(bT,csmT,maskT):
    tmp=maskT*bT
    tmp=sf.pt_ifft2c(tmp)
    csmConj=sf.pt_conj(csmT)
    for i in range(NF):
        tmp1=sf.pt_cpx_multipy(csmConj, tmp[i])
        atbT[i]=torch.sum( tmp1 ,dim=-4)
    return atbT


#%%
    
x = shepp_logan_phantom().astype(np.complex)
im_size = x.shape
x = np.stack((np.real(x), np.imag(x)))
# convert to tensor, unsqueeze batch and coil dimension
# output size: (1, 1, 2, ny, nx)
x = torch.tensor(x).unsqueeze(0).unsqueeze(0)

klength = 64
ktraj = np.stack(
    (np.zeros(64), np.linspace(-np.pi, np.pi, klength))
)
# convert to tensor, unsqueeze batch dimension
# output size: (1, 2, klength)
ktraj = torch.tensor(ktraj).unsqueeze(0)

nufft_ob = KbNufft(im_size=im_size)
# outputs a (1, 1, 2, klength) vector of k-space data
kdata = nufft_ob(x, ktraj)
#%% generate atb
sigma=0.0
lam=1e-5
cgIter=10
cgTol=1e-15


A=lambda x: pt_A(x,csmT,maskT)
At=lambda x: pt_At(x,csmT,maskT)

bT=At(orgT)
noiseT=torch.randn(bT.shape)*sigma
noiseT=noiseT.to(gpu)
bT=bT+noiseT
atbT=orgT#At(bT)

#%% run cg-sense
B=lambda x: At(A(x))+lam*x
sf.tic()
recT=sf.pt_cg(B,atbT,cgIter,cgTol)
sf.toc()
#%%
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