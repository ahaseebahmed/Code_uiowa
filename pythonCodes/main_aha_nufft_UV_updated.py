"""
This is the UV-storm code in Pytorch using NUFFT 
@author: abdul haseeb
"""
from __future__ import division
import numpy as np
import numpy.matlib as mt
import matplotlib.pyplot as plt
import torch
import supportingFun as sf
import h5py as h5
from torchkbnufft.torchkbnufft import KbNufft
from torchkbnufft.torchkbnufft import AdjKbNufft
from torchkbnufft.torchkbnufft import MriSenseNufft
from torchkbnufft.torchkbnufft import AdjMriSenseNufft
from torchkbnufft.torchkbnufft.nufft.sparse_interp_mat import precomp_sparse_mats
from espirit.espirit import espirit, espirit_proj, ifft
import scipy.io as sio
from scipy.linalg import fractional_matrix_power


#%% initiazationa parameters
gpu=torch.device('cuda:0')
directory='/Users/ahhmed/pytorch_sense/'
dtype = torch.float

NF=30 
nx=340
nch=4
nkpts=2336
nintl=6
nbasis=30
lam=500
#%% Loading and Reading data
#
## loading coil sensitivity map
#with h5.File(directory+'csm_se17.h5', 'r') as f:  
#  # coil sensitivity maps
#  csm=np.asarray(f['/csm/re'])+np.asarray(f['/csm/im'])*1j
#  csm=csm.astype(np.complex64)
#  #csmTrn=np.transpose(csm,[0,2,1])
#  #csm=np.fft.fftshift(np.fft.fftshift(csm,1),2)
#  csmTrn=csm[0:nch,:,:]
#  ncoils=csmTrn.shape[0]
#  del csm
#
## loading k-space data
#with h5.File(directory+'kdata_se17.h5', 'r') as f:  
#  # reading the RHS: AHb
#  kdata=np.asarray(f['/kdata/re'])
#  kdata=kdata+np.asarray(f['/kdata/im'])*1j
#  kdata=kdata.astype(np.complex64)
#  kdata=np.squeeze(kdata[0:nch,0:NF,:])
#  kdata=np.reshape(kdata,(nch,NF*nintl*nkpts))
##kdata=np.reshape(kdata,(nch,NF*6*2336)) 
#
##loading density compensation function
#with h5.File(directory+'dcf.h5', 'r') as f:  
#  # reading the RHS: AHb
#  dcf=np.asarray(f['/dcf/re'])
#  dcf=np.reshape(dcf[0:NF*nintl],(1,NF*nintl*nkpts))
#  dcf=np.tile(dcf,(nbasis,nch,2,1,1))
#
##loading k-space trajectory
#with h5.File(directory+'ktraj_se17.h5', 'r') as f: 
#  ktraj=np.asarray(f['/csm/re'])
#  ktraj=ktraj+np.asarray(f['/csm/im'])*1j
#  ktraj=ktraj.astype(np.complex64)
#  ktraj=np.squeeze(np.transpose(ktraj[0:NF,:],[0,1]))*2*np.pi
#ktraj=np.reshape(ktraj,(1,NF*nintl*nkpts))  
#
##loading Laplacian matrix
#with h5.File(directory+'L.h5', 'r') as f:  
#  # reading the RHS: AHb
#  L=np.asarray(f['/L'])
#  U,sbasis,V=np.linalg.svd(L)
#  V=V[NF-nbasis:NF,:]
#  sbasis=np.diag(sbasis[NF-nbasis:NF,])*lam
#Vn=np.zeros((nbasis,NF,NF))  
#for i in range(nbasis):
#    Vn[i,:,:]=np.diag(V[i,:])
#
       
dictn=sio.loadmat('./../series81.mat')
kdata=np.asarray(dictn['kdata'])
kdata=kdata.astype(np.complex64)
kdata=np.transpose(kdata,(1,2,0))
kdata=kdata[0:nch,0:NF*nintl,:]
kdata=np.reshape(kdata,(nch,NF*nintl*nkpts))

ktraj=np.asarray(dictn['k'])
ktraj=ktraj.astype(np.complex64)
ktraj=ktraj[:,0:NF*nintl]
ktraj=np.squeeze(np.transpose(ktraj,[1,0]))*2*np.pi
ktraj=np.reshape(ktraj,(1,NF*nintl*nkpts))  

dcf=np.asarray(dictn['dcf'])
#%% Coil combine
thres=0.8
Rs=np.real(kdata@np.transpose(np.conj(kdata)))
[w,v]=np.linalg.eig(Rs)
#ind=np.argsort(w)
#w=w[ind]
#v=v[:,ind]
w=w/sum(w)
w=np.cumsum(w)
nvch=np.min(np.where(w>thres))
#vkdata=np.transpose(v[:,0:nvch[0].item()])@kdata
vkdata=np.transpose(v[:,0:nvch])@kdata

nch=vkdata[:,0].size
#%%Laplacian estimation
sigsq=4.5
lam1=0.1
tmp=np.reshape(vkdata,(nch,NF,nintl,nkpts))
nav=tmp[:,:,0,:] 
nav=np.transpose(nav,(2,0,1))
nav=np.reshape(nav,(nch*nkpts,NF))
del tmp
q,eta=-0.5,2
x2=np.sum((nav*np.conj(nav)),axis=0)
x2=x2[np.newaxis]
x3=np.transpose(np.conj(nav))@nav
dsq=np.abs(mt.repmat(x2,NF,1)+mt.repmat(np.transpose(np.conj(x2)),1,NF))+2*np.real(x3)
med=np.median(np.reshape(dsq,-1))
nav=nav/np.sqrt(med)
dsq=dsq/med
tt=dsq/sigsq
K=np.exp(-tt)
s,u=np.linalg.eig(K)
gamma=100
S=np.diag(s)
for i in range(70):
    print(i)
    tmp=fractional_matrix_power((S+gamma*np.eye(NF)),-q)
    w=u@tmp@np.transpose(np.conj(u))
    A=w*K
    A=A+np.diag(np.sum(A,axis=0))
    X=nav@np.linalg.inv(np.eye(NF)+lam1*A)
    gamma=gamma/eta
    x2=np.sum((nav*np.conj(nav)),axis=0)
    x2=x2[np.newaxis]
    x3=np.transpose(np.conj(nav))@nav
    dsq=np.abs(mt.repmat(x2,NF,1)+mt.repmat(np.transpose(np.conj(x2)),1,NF))+2*np.real(x3)
    tt=dsq/sigsq
    K=np.exp(-tt)
    s,u=np.linalg.eig(K)
    S=np.diag(s)
A=A.astype(np.complex64)
U,sbasis,V=np.linalg.svd(A)
sbasis[NF-1]=0
V=V[NF-nbasis:NF,:]
sbasis=np.diag(sbasis[NF-nbasis:NF,])*lam
V=V.astype(np.float32)
Vn=np.zeros((nbasis,NF,NF))  
for i in range(nbasis):
    Vn[i,:,:]=np.diag(V[i,:])

#%%
im_size = [nx,nx]#csmTrn[0].shape
kdata = np.stack((np.real(vkdata), np.imag(vkdata)),axis=1)

# convert to tensor, unsqueeze batch and coil dimension
kdataT = torch.tensor(kdata).to(dtype).unsqueeze(0)
#dcfT = torch.tensor(dcf).to(dtype).squeeze(3)
VT = torch.tensor(Vn).to(dtype)
sbasis = torch.tensor(sbasis).to(dtype)
# convert to tensor, unsqueeze batch dimension
# output size: (1, 2, klength)
ktraj = np.stack((np.real(ktraj), np.imag(ktraj)), axis=1)
ktraj=np.tile(ktraj,(nbasis,1,1))
ktrajT = torch.tensor(ktraj).to(dtype)

#%% convert them to gpu
kdataT=kdataT.to(gpu)
ktrajT=ktrajT.to(gpu)
#dcfT=dcfT.to(gpu)
VT=VT.to(gpu)
sbasis=sbasis.to(gpu)

#%%% Estimating coil images and coil senstivity maps
nuf_ob = KbNufft(im_size=im_size).to(dtype)
nuf_ob=nuf_ob.to(gpu)
adjnuf_ob = AdjKbNufft(im_size=im_size).to(dtype)
adjnuf_ob=adjnuf_ob.to(gpu)

coilimages=torch.zeros((1,nch,2,nx,nx))
A=lambda x: nuf_ob(x,ktrajT)
At=lambda x:adjnuf_ob(x,ktrajT)
AtA=lambda x:At(A(x))
for i in range(nch):
    temp=adjnuf_ob(kdataT[:,i].unsqueeze(1),ktrajT[0].unsqueeze(0))
    ini=torch.zeros_like(temp)
    coilimages[:,i]=sf.pt_cg(AtA,temp,ini,50,1e-15)
X=coilimages.cpu().numpy()
x=X[:,:,0]+X[:,:,1]*1j
x=np.transpose(x,(0,2,3,1))
x_f = ifft(x, (0, 1, 2))
csmTrn = espirit(x_f, 6, 24, 0.1, 0.9925)
csm=csmTrn[0,:,:,:,0]
csm=np.transpose(csm,(2,0,1))
smap = np.stack((np.real(csm), np.imag(csm)), axis=1)
smap=np.tile(smap,(nbasis,1,1,1,1))
smapT = torch.tensor(smap).to(dtype)
smapT=smapT.to(gpu)

#%% generate MRI sense Nufft operator 

nufft_ob = MriSenseNufft(im_size=im_size, smap=smapT).to(dtype)
nufft_ob=nufft_ob.to(gpu)
adjnufft_ob = AdjMriSenseNufft(im_size=im_size, smap=smapT).to(dtype)
adjnufft_ob=adjnufft_ob.to(gpu)

atb=torch.zeros((nbasis,im_size[0],im_size[1]))
kdataT=torch.reshape(kdataT,(1,nch,2,NF,nintl*nkpts))
kdataT=kdataT.permute(3,0,1,2,4)
kdataT=torch.reshape(kdataT,(NF,1*nch*2*nintl*nkpts))

temp=torch.matmul(VT,kdataT)
temp=torch.reshape(temp,(nbasis,NF,1,nch,2,nintl*nkpts))

temp=temp.permute(0,2,3,4,1,5)
temp=torch.reshape(temp,(nbasis,nch,2,NF*nintl*nkpts))

# generating AHb results
atb=adjnufft_ob(temp,ktrajT)
del temp, kdataT
#%% Function AtA with UV
def AtA_UV(x,vt):
    ktb=nufft_ob(x,ktrajT)
    ktb=torch.reshape(ktb,(nbasis,nch,2,NF,nintl*nkpts))
    ktb=ktb.permute(3,0,1,2,4)
    ktb=torch.reshape(ktb,(NF,nbasis,nch*2*nintl*nkpts))
    
    tmp1=torch.zeros((NF,nch*2*nintl*nkpts)).to('cuda:0')
    
    for i in range(nbasis):
        tmp1=tmp1+vt[i,]@ktb[:,i,:]
        
    vt=torch.reshape(vt,(nbasis*NF,NF))
    tmp2=VT@tmp1
    tmp2=torch.reshape(tmp2,(nbasis,NF,nch,2,nintl*nkpts))
    tmp2=tmp2.permute(0,2,3,1,4)
    tmp2=torch.reshape(tmp2,(nbasis,nch,2,NF*nintl*nkpts))
    atb1=adjnufft_ob(tmp2,ktrajT)
    x=torch.reshape(x,(nbasis,1*2*nx*nx))
    reg=sbasis.T@x
    reg=torch.reshape(reg,(nbasis,1,2,nx,nx))
    atb1=atb1+reg
    return atb1


#%% congugate gradient parameter and code to get the final images.
#lam=1e-2
cgIter=10
cgTol=1e-15

recT=torch.zeros_like(atb)
B=lambda x: AtA_UV(x,VT)
sf.tic()
recT=sf.pt_cg(B,atb,recT,cgIter,cgTol)
sf.toc()

rec = np.squeeze(recT.cpu().numpy())
rec = rec[:,0] + 1j*rec[:,1]
#%% view the results
#for i in range(30):
#    plt.imshow((np.squeeze(np.abs(rec[i,:,:]))),cmap='gray')       
#    plt.show() 
#    plt.pause(0.01)

rec=np.reshape(rec,(nbasis,nx*nx))
rec1=rec.T@V
rec1=np.reshape(rec1,(nx,nx,NF))
for i in range(50):
    plt.imshow((np.squeeze(np.abs(rec1[:,:,i]))),cmap='gray')       
    plt.show() 
    plt.pause(0.05)

