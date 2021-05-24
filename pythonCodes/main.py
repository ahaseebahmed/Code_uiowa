"""
Created on Tue Apr 14 11:36:39 2020
This is the sense code in pytorch
@author: haggarwal
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import supportingFun as sf

gpu=torch.device('cuda:0')

org=np.load('data.npz')['org']
csm=np.load('data.npz')['csm']
mask=np.load('data.npz')['mask']


#%% make them pytorch tensors
orgT=torch.tensor(sf.c2r(org))
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
    tmp=sf.pt_cpx_multipy(csmConj, tmp)
    atbT=torch.sum( tmp ,dim=-4)
    return atbT

#%% generate atb
sigma=0.0
lam=1e-5
cgIter=10
cgTol=1e-15


A=lambda x: pt_A(x,csmT,maskT)
At=lambda x: pt_At(x,csmT,maskT)

bT=A(orgT)
noiseT=torch.randn(bT.shape)*sigma
noiseT=noiseT.to(gpu)
bT=bT+noiseT
atbT=At(bT)

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