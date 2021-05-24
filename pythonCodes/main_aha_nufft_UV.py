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
nch=2
nkpts=2336
nintl=6
nbasis=30
lam=0.01
#%%

with h5.File(directory+'csm_se17.h5', 'r') as f:  
  # coil sensitivity maps
  csm=np.asarray(f['/csm/re'])+np.asarray(f['/csm/im'])*1j
  csm=csm.astype(np.complex64)
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
  kdata=np.squeeze(kdata[0:nch,0:NF,:])
  kdata=np.reshape(kdata,(nch,NF*nintl*nkpts))
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
  dcf=np.reshape(dcf[0:NF*nintl],(1,NF*nintl*nkpts))
  dcf=np.tile(dcf,(nbasis,nch,2,1,1))

with h5.File(directory+'ktraj_se17.h5', 'r') as f: 
  ktraj=np.asarray(f['/csm/re'])
  ktraj=ktraj+np.asarray(f['/csm/im'])*1j
  ktraj=ktraj.astype(np.complex64)
  ktraj=np.squeeze(np.transpose(ktraj[0:NF,:],[0,1]))*2*np.pi
ktraj=np.reshape(ktraj,(1,NF*nintl*nkpts))  


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

kdata = np.stack((np.real(kdata), np.imag(kdata)),axis=1)
# convert to tensor, unsqueeze batch and coil dimension
# output size: (1, 1, 2, ny, nx)
kdataT = torch.tensor(kdata).to(dtype).unsqueeze(0)

dcfT = torch.tensor(dcf).to(dtype).squeeze(3)
LT = torch.tensor(L).to(dtype)
VT = torch.tensor(Vn).to(dtype)
sbasis = torch.tensor(sbasis).to(dtype)

smap = np.stack((np.real(csmTrn), np.imag(csmTrn)), axis=1)
smap=np.tile(smap,(nbasis,1,1,1,1))
smapT = torch.tensor(smap).to(dtype)
# convert to tensor, unsqueeze batch dimension
# output size: (1, 2, klength)
ktraj = np.stack((np.real(ktraj), np.imag(ktraj)), axis=1)
ktraj=np.tile(ktraj,(nbasis,1,1))
ktrajT = torch.tensor(ktraj).to(dtype)

#y = np.stack((np.real(y), np.imag(y)), axis=0)
#yT = torch.tensor(y).to(dtype).unsqueeze(0).unsqueeze(0)

#%% take them to gpu
kdataT=kdataT.to(gpu)
smapT=smapT.to(gpu)
ktrajT=ktrajT.to(gpu)
dcfT=dcfT.to(gpu)
#LT=LT.to(gpu)
VT=VT.to(gpu)
sbasis=sbasis.to(gpu)
#%% generate atb
sigma=0.0
lam=1e-2
cgIter=1
cgTol=1e-15

#nuf_ob = KbNufft(im_size=im_size).to(dtype)
#nuf_ob=nuf_ob.to(gpu)
#adjnuf_ob = AdjKbNufft(im_size=im_size).to(dtype)
#adjnuf_ob=adjnuf_ob.to(gpu)
#real_mat, imag_mat = precomp_sparse_mats(ktrajT, adjnuf_ob)
#interp_mats = {'real_interp_mats': real_mat, 'imag_interp_mats': imag_mat}

nufft_ob = MriSenseNufft(im_size=im_size, smap=smapT).to(dtype)
nufft_ob=nufft_ob.to(gpu)
adjnufft_ob = AdjMriSenseNufft(im_size=im_size, smap=smapT).to(dtype)
adjnufft_ob=adjnufft_ob.to(gpu)
#nufft_ob = MriSenseNufft(smap=smapT,im_size=im_size).to(dtype)
#adjnufft_ob = AdjMriSenseNufft(smap=smapT,im_size=im_size ).to(dtype)

#At=lambda x: adjnufft_ob(x*dcfT,ktrajT,interp_mats)
#A=lambda x: nufft_ob(x,ktrajT,interp_mats)

atb=torch.zeros((nbasis,im_size[0],im_size[1]))
kdataT=torch.reshape(kdataT,(1,nch,2,NF,nintl*nkpts))
kdataT=kdataT.permute(3,0,1,2,4)
kdataT=torch.reshape(kdataT,(NF,1*nch*2*nintl*nkpts))

temp=torch.matmul(VT,kdataT)
temp=torch.reshape(temp,(nbasis,NF,1,nch,2,nintl*nkpts))

temp=temp.permute(0,2,3,4,1,5)
temp=torch.reshape(temp,(nbasis,nch,2,NF*nintl*nkpts))
atb=adjnufft_ob(temp*dcfT,ktrajT)
del temp
del kdataT

#ktb=nufft_ob(atb,ktrajT)
#ktb=torch.reshape(ktb,(nbasis,nch,2,NF,nintl*nkpts))
#ktb=ktb.permute(3,0,1,2,4)
#ktb=torch.reshape(ktb,(NF,nbasis,nch*2*nintl*nkpts))
#
#tmp1=torch.zeros((NF,nch*2*nintl*nkpts)).to(gpu)
#for i in range(nbasis):
#    tmp1=tmp1+torch.matmul(VT[i,],ktb[:,i,:])
#    
#tmp2=torch.matmul(VT,tmp1)
#tmp2=torch.reshape(tmp2,(nbasis,NF,nch,2,nintl*nkpts))
#tmp2=tmp2.permute(0,2,3,1,4)
#tmp2=torch.reshape(tmp2,(nbasis,nch,2,NF*nintl*nkpts))
#atb1=adjnufft_ob(tmp2,ktrajT)


#AtA=lambda x: sf.AtA_UV(nufft_ob,adjnufft_ob,x,ktrajT,VT,im_size,nbasis,NF,nch,nintl,nkpts,dcfT)
#A=lambda x: nufft_ob(x,ktrajT)
#kdataT1=A(yT)
#atbT=At(kdataT)
#del kdataT
#xx=sf.complex_multiplication(atbT,LT)
#xx=xx.to(gpu)

def AtA_UV(x):
    
    ktb=nufft_ob(x,ktrajT)
    ktb=torch.reshape(ktb,(nbasis,nch,2,NF,nintl*nkpts))
    ktb=ktb.permute(3,0,1,2,4)
    ktb=torch.reshape(ktb,(NF,nbasis,nch*2*nintl*nkpts))
    
    tmp1=torch.zeros((NF,nch*2*nintl*nkpts)).to('cuda:0')
    
    for i in range(nbasis):
        tmp1=tmp1+VT[i,]@ktb[:,i,:]
        
    VT=torch.reshape(VT,(nbasis*NF,NF))
    tmp2=VT@tmp1
    tmp2=torch.reshape(tmp2,(nbasis,NF,nch,2,nintl*nkpts))
    tmp2=tmp2.permute(0,2,3,1,4)
    tmp2=torch.reshape(tmp2,(nbasis,nch,2,NF*nintl*nkpts))
    atb1=adjnufft_ob(tmp2*dcfT,ktrajT)
    x=torch.reshape(x,(nbasis,1*2*nx*nx))
    reg=sbasis.T@x
    atb1=atb1+reg
    return atb1


#%% run cg-sense
recT=torch.zeros_like(atb)
B=lambda x: AtA(x)
sf.tic()
recT=sf.pt_cg(B,atb,recT,cgIter,cgTol)
sf.toc()

rec = np.squeeze(recT.cpu().numpy())
rec = rec[:,0] + 1j*rec[:,1]
#%%
for i in range(30):
    plt.imshow((np.squeeze(np.abs(rec[i,:,:]))),cmap='gray')       
    plt.show() 
    plt.pause(0.01)

rec1=np.matmul(rec.T,V)
for i in range(100):
    plt.imshow((np.squeeze(np.abs(rec1[:,:,i]))),cmap='gray')       
    plt.show() 
    plt.pause(0.05)

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
