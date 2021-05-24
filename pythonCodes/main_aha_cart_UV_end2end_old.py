"""
Created on Tue Apr 14 11:36:39 2020
This is the cart_UV code in pytorch
@author: abdul haseeb
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import supportingFun as sf
import h5py as h5
import readData as rd
import scipy.io as sio
import sys
sys.path.insert(0, "/Users/ahhmed/pytorch_unet")

from torchkbnufft.torchkbnufft import KbNufft
from torch.utils.data import Dataset, DataLoader
from model import UnetClass
from spatial_model import SmallModel
from scipy import ndimage

gpu=torch.device('cuda:0')
directory='/Users/ahhmed/Codes/TensorFlow_SToRM/Data1/FID142454v/d900/'
dtype = torch.float
directory1='/Users/ahhmed/pytorch_unet/SPTmodels/28Apr_030506pm_41ep_noise_1/'
chkPoint='41'

NF=200 
nx=512
#N=400
N1=300
N2=50
nch=2
thres=0.05
nbasis=10
lam=0.05
v=np.zeros((nbasis,nbasis,NF))
#%%

with h5.File(directory+'L.h5','r') as f:
  # load the L matrix
  L=np.asarray(f['L']) 
  L=L[0:NF,0:NF]
  L=L.astype(np.complex64)
  U,sbasis,V=np.linalg.svd(L)
  sbasis[NF-1]=0
  V=V[NF-nbasis:NF,:]
  sbasis=np.diag(sbasis[NF-nbasis:NF,])*lam
  V=V.astype(np.float32)
#  V=np.expand_dims(V,axis=1)
#  for i in range(NF):
#      v[:,:,i]=V[:,:,i]@V[:,:,i].T
#  V=np.squeeze(V)
#Vn=np.zeros((nbasis,NF,NF))  
#for i in range(nbasis):
#    Vn[i,:,:]=np.diag(V[i,:])
with h5.File(directory+'coilimages.h5', 'r') as f:  
  # coil sensitivity maps
  cimg=np.asarray(f['/coilimages/re'])+np.asarray(f['/coilimages/im'])*1j
  cimg=cimg.astype(np.complex64)
  cimg=np.transpose(cimg,[0,2,1])
  #cimg=np.fft.fftshift(np.fft.fftshift(cimg,1),2)
  cimg=np.sum(np.abs(cimg),axis=0)
  cimg1=ndimage.convolve(cimg,np.ones((13,13)))
  W=cimg1/np.max(np.abs(cimg1))
  W[W<thres]=0;
  W[W>=thres]=1;
  W=W.astype(int)
  
#W1=ndimage.morphology.binary_closing(W,structure=np.ones((3,3)))
W1=ndimage.morphology.binary_fill_holes(W)
W1=W1.astype(float)
W1=ndimage.convolve(W1,np.ones((13,13)))
W1=W1/np.max(W1)
W1=(1-W1)
W1=1+W1*10
W1=np.reshape(W1,(1,nx*nx))
#W1=np.multiply(W1,W1)
W=np.tile(W1,(nbasis,2,1))
W=np.transpose(W,(0,2,1))
W=np.reshape(W,(nbasis,nx*nx*2))
#W=1+W*10
#W=W*10
#cimg=np.tile(cimg,(NF,1))
#cimg=np.reshape(cimg,(NF,nx,nx))  
del cimg1 
del cimg

with h5.File(directory+'csm.h5', 'r') as f:  
  # coil sensitivity maps
  csm=np.asarray(f['/csm/re'])+np.asarray(f['/csm/im'])*1j
  csm=csm.astype(np.complex64)
  csm=csm[0:nch,:,:]
  csmTrn=np.transpose(csm,[0,2,1])
  #csm=np.fft.fftshift(np.fft.fftshift(csm,1),2)
  
  ncoils=csmTrn.shape[0]
  
with h5.File(directory+'kdata.h5', 'r') as f:  
  # reading the RHS: AHb
  kdata=np.asarray(f['/kdata/re'])
  kdata=kdata+np.asarray(f['/kdata/im'])*1j
  kdata=kdata.astype(np.complex64)
  #atb=np.squeeze(np.transpose(atb[0:NF,:,:],[0,2,1]))  
  #atb=np.fft.fftshift(np.fft.fftshift(atb,1),2)

tmp1=np.zeros((nx,nx))
atbV=np.zeros((nx*nx,nbasis),dtype=complex)  
for i in range(NF):
    for j in range(nch):
        tmp=kdata[j,i,:]
        tmp=np.expand_dims(tmp,axis=1)
        tmp=np.reshape(tmp,(nx,nx))
        tmp1=tmp1+np.fft.ifft2(tmp)*np.conj(csm[j])
        
    tmp2=np.expand_dims(V[:,i],axis=1)
    tmp3=np.reshape(tmp1,(nx*nx,1))
    atbV=atbV+np.matmul(tmp3,tmp2.T)
    tmp1=np.zeros((nx,nx))

atbV=atbV*nx
atbV=np.transpose(atbV,[1,0])
atbV=np.reshape(atbV,(nbasis,nx,nx))
atbV=atbV.astype(np.complex64)
atbV=np.transpose(atbV,[0,2,1])
del tmp,tmp1,tmp2,tmp3

with h5.File(directory+'S_mask.h5', 'r') as f: 
  maskTst=np.asarray(f['S'])
  maskTst=np.squeeze(np.transpose(maskTst[0:NF,:,:],[0,1,2]))
  maskTst=np.tile(maskTst[...,np.newaxis],(1,1,1,2))
  maskTst=np.reshape(maskTst,(NF,nx*nx*2))

#tmp2=np.zeros((nx*nx*2,nbasis,nbasis))
#for i in range(NF):
#    tmp=(np.diag(V[:,i])@np.tile(maskTst[i],(nbasis,1)))
#    tmp1=np.matmul(np.expand_dims(tmp,axis=0).T,np.expand_dims(V[:,i],axis=1).T)
#    tmp2=tmp2+tmp1
#    res=res+tmp1

#%% make them pytorch tensors
atbT=torch.tensor(sf.c2r(atbV))
csmT=torch.tensor(sf.c2r(csmTrn))
maskT=torch.tensor(maskTst)
VT=torch.tensor(V)
sT=torch.tensor(sbasis)
W=torch.tensor(W)

#%% take them to gpu
torch.cuda.empty_cache()
atbT=atbT.to(gpu)
csmT=csmT.to(gpu)
maskT=maskT.to(gpu,dtype)
VT=VT.to(gpu)
W=W.to(gpu,dtype)
sT=sT.to(gpu)
#%% creating the training model
#unet=UnetClass()
#unet=unet.to(gpu)

unet=SmallModel()
unet=unet.to(gpu)

optimizer=torch.optim.Adam(unet.parameters(),lr=1e-3)
def lossFun(pred,org):
    loss=torch.mean(torch.abs(pred-org))
    return loss

#%% creating validation model
directory2='/Users/ahhmed/pytorch_sense'
ckpt=400
unet=SmallModel()
modelDir=wtsFname=directory2+'/wts-'+str(ckpt)+'.pt'
unet.load_state_dict(torch.load(modelDir))
unet.eval()
unet.to(gpu)
#%% make A and At operators   
def AtAUV(x,csmT,csmConj,maskT):
    atbv=torch.cuda.FloatTensor(nbasis,nx,nx,2).fill_(0)

    for i in range(nch):
        tmp2=sf.pt_fft2c(sf.pt_cpx_multipy(x,csmT[i].repeat(nbasis,1,1,1)))
        tmp2=torch.reshape(tmp2,(nbasis,nx*nx*2))
        tmp2=tmp2.repeat(nbasis,1,1)*maskT
        tmp=tmp2.sum(axis=1)
        tmp=torch.reshape(tmp,(nbasis,nx,nx,2))
        tmp2=sf.pt_cpx_multipy(csmConj[i].repeat(nbasis,1,1,1),sf.pt_ifft2c(tmp))
        atbv=atbv+tmp2
        del tmp2
    x=torch.reshape(x,(nbasis,nx*nx*2))
    x=W*x
    reg=torch.mm(sT,x)
    reg=torch.reshape(reg,(nbasis,nx,nx,2))
    atbv=atbv+reg
    return atbv
#%%
def rhs(u1,D):
    y=[]
    u1=torch.reshape(u1,(nbasis,nx*nx*2))
    x=torch.mm(D.T,u1)
    x=torch.reshape(x,(10,nx,nx,2))
    x=x.permute(0,3,1,2)
    trnDs=rd.DataGen(x.detach().cpu().numpy())
    ldr=DataLoader(trnDs,batch_size=1,shuffle=True,num_workers=8,pin_memory=True)
    for i, data in enumerate(ldr):
        data=data.to(gpu,dtype)
        slcR=unet(data)
        y.append(slcR)
    Y=torch.cat(y)
    Y=Y.permute(0,2,3,1)
    Y=torch.reshape(Y,(10,nx*nx*2))
    z=torch.mm(D,Y)
    z=torch.reshape(z,(nbasis,nx,nx,2))
    return z

#%% create model and load the weights
def reg_term(u1,D):
    u=torch.reshape(u1,(nbasis,nx*nx*2))
    x=torch.mm(D.T,u)
    u2=torch.mm(D,x)
    u2=torch.reshape(u2,(nbasis,nx,nx,2))
    return u2 
#%%
lam2=1e-3
cgIter=1
cgTol=1e-15
out_iter=3
cgIter1=10
epochs=400
csmConj=sf.pt_conj(csmT)
res=torch.cuda.FloatTensor(nbasis,nbasis,nx*nx*2).fill_(0)

for k in range(NF):
    tmp3=maskT[k].repeat(nbasis,1)#*tmp2
    tmp=torch.diag_embed(VT[:,k])@tmp3#np.tile(maskTst[i],(nbasis,1)))
    #tmp1=torch.diag_embed(VT[:,i])
    tmp=tmp.unsqueeze(0)
    tmp=tmp.repeat(nbasis,1,1)
    tmp=torch.reshape(tmp,(nbasis,nbasis*nx*nx*2))
    tmp1=torch.diag_embed(VT[:,k])@tmp
    tmp1=torch.reshape(tmp1,(nbasis,nbasis,nx*nx*2))
    #tmp1=np.matmul(np.expand_dims(tmp,axis=0).T,np.expand_dims(V[:,i],axis=1).T)
    #tmp7=tmp1.sum(axis=0)
    #tmp1=tmp1.sum(axis=1)
    res=res+tmp1

#AtA1=lambda x: AtAUV(x,csmT,maskT)+lam2*reg_term1(x,VT,NF,lam2)
#%% run method
indx=torch.randint(0,NF,(10,))
#indx=0:NF
#indx=torch.tensor(indx)
epLoss=0;
vt=VT[:,indx]
recT=torch.zeros_like(atbT)
AtA=lambda x: AtAUV(x,csmT,csmConj,res)
recT=sf.pt_cg(AtA,atbT,atbT,80,cgTol)
AtA=lambda x: AtAUV(x,csmT,csmConj,res)+lam2*reg_term(x,vt)
      
for ep in range(epochs):    
    #sf.tic()
    for i in range(out_iter):
        atbT1=atbT+lam2*rhs(recT,vt)
        recT=sf.pt_cg(AtA,atbT1,recT,4,cgTol)

    #sf.toc()
    loss=lossFun(atbT1,atbT)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    epLoss+=loss.item()

if ((ep+1) % 50==0) or ((ep+1)==epochs) :
    wtsFname=directory2+'/wts-'+str(ep+1)+'.pt'
    torch.save(unet.state_dict(), wtsFname)



for param in unet.parameters():
    print(param.grad.data)
    


#%%
indx=torch.randint(0,NF,(10,))

vt=VT[:,indx]
AtA=lambda x: AtAUV(x,csmT,maskT)+lam2*reg_term(x,vt)      
recT=torch.zeros_like(atbT)
atbT1=atbT
#sf.tic()
for i in range(out_iter):
    recT=sf.pt_cg(AtA,atbT1,recT,cgIter,cgTol)
    atbT1=atbT+lam2*rhs(recT,vt)
#%%
#sf.tic()
#recT,err=sf.pt_cg(AtA1,atbT,atbT,cgIter1,cgTol)
#sf.toc()
#%%
rec = np.squeeze(recT.detach().cpu().numpy())
rec = rec[:,:,:,0] + 1j*rec[:,:,:,1]
#for i in range(30):
#    plt.imshow((np.squeeze(np.abs(xx[27,:,:]))),cmap='gray')       
#    plt.show() 
#    plt.pause(1)
#xx=np.fft.fftshift(np.fft.fftshift(atbV,1),2)

rec1=np.matmul(rec.T,V)
rec1=np.fft.fftshift(np.fft.fftshift(rec1,0),1)
#plt.figure(1,[15,15])
for i in range(50):
    plt.imshow((np.squeeze(np.abs(rec1[:,:,i]))),cmap='gray')       
    plt.show() 
    plt.pause(0.05)

#sio.savemat('temp_prp400.mat',{"rec":rec1}) 
    
#psnrAtb=sf.myPSNR(normOrg,normAtb)
##ssimAtb=sf.mySSIM(normOrg,normAtb)
#
#psnrRec=sf.myPSNR(normOrg,normRec)
##ssimRec=sf.mySSIM(normOrg,normRec)
#print ('  ' + 'Noisy ' + 'Rec')
#print ('  {0:.2f} {1:.2f}'.format(psnrAtb.mean(),psnrRec.mean()))
##print ('  {0:.2f} {1:.2f}'.format(ssimAtb.mean(),ssimRec.mean()))
##%% view the results
#
#fig,ax=plt.subplots(1,3)
#ax[0].imshow(normOrg)
#ax[0].set_title('original')
#ax[0].axis('off')
#ax[1].imshow(normAtb)
#ax[1].set_title('Atb ')
#ax[1].axis('off')
#ax[2].imshow(normRec)
#ax[2].set_title('Recon ')
#ax[2].axis('off')
#plt.show()
#
#for i in range(30):
#    plt.imshow((np.squeeze(np.abs(normAtb[i,:,:]))),cmap='gray')       
#    plt.show() 
#    plt.pause(0.1)