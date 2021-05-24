"""
Created on Tue Apr 14 11:36:39 2020
This is the cart_UV code in pytorch
@author: abdul haseeb
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import supportingFun as sf
from torch.autograd import Variable
import h5py as h5
import readData as rd
import scipy.io as sio
import sys
from torch.utils.data import TensorDataset

sys.path.insert(0, "/Users/ahhmed/pytorch_unet")

from torchkbnufft.torchkbnufft import KbNufft
from torch.utils.data import Dataset, DataLoader
#from model import UnetClass
from spatial_model6 import SmallModel

from scipy import ndimage

gpu=torch.device('cuda:0')
cpu=torch.device('cpu')
directory='/Users/ahhmed/Codes/TensorFlow_SToRM/Data1/FID142454v/d900/'
dtype = torch.float
directory1='/Users/ahhmed/pytorch_unet/SPTmodels/19Jun_101938pm_50ep_20Jun/'
#directory1='/Users/ahhmed/pytorch_unet/SPTmodels/21Jun_110930pm_100ep_21Jun/'
#directory1='/Users/ahhmed/pytorch_unet/SPTmodels/23Jun_111055pm_150ep_23Jun/'

#chkPoint='24'
chkPoint='52'


NF=200#900 
nx=512
#N=400
N1=300
N2=50
nch=2
thres=0.05
nbasis=30
lam=0.05
st=210#0
#%%

with h5.File(directory+'L.h5','r') as f:
  # load the L matrix
  L=np.asarray(f['L']) 
  L=L[0+st:NF+st,0+st:NF+st]
  L=L.astype(np.complex64)
  U,sbasis,V=np.linalg.svd(L)
  sbasis[NF-1]=0
  V=V[NF-nbasis:NF,:]
  sbasis=np.diag(sbasis[NF-nbasis:NF,])*lam
  V=V.astype(np.float32)
#Vn=np.zeros((nbasis,NF,NF))  
#for i in range(nbasis):
#    Vn[i,:,:]=np.diag(V[i,:])
with h5.File(directory+'coilimages.h5', 'r') as f:  
  # coil sensitivity maps
  cimg=np.asarray(f['/coilimages/re'])+np.asarray(f['/coilimages/im'])*1j
  cimg=cimg.astype(np.complex64)
  #cimg=np.fft.fftshift(np.fft.fftshift(cimg,1),2)
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
  #csm1=np.fft.fftshift(np.fft.fftshift(csm,1),2)
  csmTrn=np.transpose(csm,[0,1,2])
  #csm=np.fft.fftshift(np.fft.fftshift(csm,1),2)
  
  ncoils=csmTrn.shape[0]
  
with h5.File(directory+'kdata.h5', 'r') as f:  
  # reading the RHS: AHb
  kdata=np.asarray(f['/kdata/re'])
  kdata=kdata+np.asarray(f['/kdata/im'])*1j
  kdata=kdata[0:nch,0+st:NF+st,:]
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
#atbV=np.fft.fftshift(np.fft.fftshift(atbV,1),2)
atbV=np.transpose(atbV,[0,2,1])
del tmp,tmp1,tmp2,tmp3

with h5.File(directory+'S_mask.h5', 'r') as f: 
  maskTst=np.asarray(f['S'])
  #maskTst=np.fft.fftshift(np.fft.fftshift(maskTst,1),2)
  maskTst=np.squeeze(np.transpose(maskTst[0+st:NF+st,:,:],[0,1,2]))
  maskTst=np.tile(maskTst[...,np.newaxis],(1,1,1,2))
  maskTst=np.reshape(maskTst,(NF,nx*nx*2))

res=np.zeros((NF,nbasis,nx*nx*2))
for k in range(NF):
    tmp3=np.tile(maskTst[k],(nbasis,1))#*tmp2
    res[k]=np.diag(V[:,k])@tmp3

#%% make them pytorch tensors
B=torch.tensor(sf.c2r(kdata))
atbT=torch.tensor(sf.c2r(atbV))
csmT=torch.tensor(sf.c2r(csmTrn))
maskT=torch.tensor(maskTst)
VT=torch.tensor(V)
sT=torch.tensor(sbasis)
W=torch.tensor(W)
#%% take them to gpu
B=B.to(cpu)
atbT=atbT.to(cpu)
csmT=csmT.to(cpu)
maskT=maskT.to(cpu,dtype=torch.bool)
VT=VT.to(cpu)
W=W.to(cpu,dtype)
sT=sT.to(cpu)
#%%
#unet=UnetClass()
#unet=SmallModel()
#modelDir=directory1+'wts-'+str(chkPoint)+'.pt'
#unet.load_state_dict(torch.load(modelDir))
#unet.eval()
#unet.to(gpu)
#%% make A and At operators   
class AUV(nn.Module):
    def __init__(self,csmT,maskT,nbasis,VT):
        super(AUV,self).__init__()
        self.csmT=csmT
        self.mask=maskT
        self.nbasis=nbasis
        self.NF=maskT.shape[0]
        self.nch=csmT.shape[0]
        self.nx=csmT.shape[1]
        self.NX=self.nx*self.nx*2
        self.VT=VT
        #self.tmp2=torch.FloatTensor(self.nch,self.nbasis,self.nx,self.nx,2).fill_(0)
        #self.tmp5=torch.FloatTensor(self.NF,self.nch*self.nx*self.nx*2).fill_(0)
                    
    def forward(self,x):
        #tmp2=torch.FloatTensor(self.nch,self.nbasis,self.nx,self.nx,2).fill_(0)
        tmp5=torch.FloatTensor(self.nch,self.NF,self.nx*self.nx*2).fill_(0)
        x=torch.reshape(x,(self.nbasis,self.nx,self.nx,2))
        for i in range(self.nch):
            tmp2=sf.pt_fft2c(sf.pt_cpx_multipy(x,self.csmT[i].repeat(self.nbasis,1,1,1)))
            tmp2=torch.reshape(tmp2,(self.nbasis,self.NX))
            for j in range(self.NF):
                tmp3=self.mask[j].repeat(nbasis,1)*tmp2
                tmp3=tmp3.T@self.VT[:,k].unsqueeze(1)
                tmp5[i,j]=tmp3.T

        return tmp5

#%%
#res=torch.cuda.FloatTensor(NF,nbasis,nx*nx*2).fill_(0)
#for k in range(NF):
#    tmp3=maskT[k].repeat(nbasis,1)#*tmp2
#    res[k]=torch.diag_embed(VT[:,k])@tmp3
    
#B=B.permute(1,0,2,3)
B=torch.reshape(B,(nch,NF,nx*nx*2))
F=AUV(csmT,maskT,nbasis,VT)
U = torch.zeros((nbasis,nx*nx*2),requires_grad=True)#,device=gpu, dtype=dtype)
U=atbT
U=Variable(U,requires_grad=True)
optimizer=torch.optim.Adam([{'params':U,'lr':1e-1}])
for t in range(100):
    b_est=F(U)
    loss=0.5*(b_est-B).pow(2).sum()
    if t%10==0:
        print(t,loss.item())
        
    #optimizer.zero_grad()
    loss.backward()
    #optimizer.step()
    with torch.no_grad():
        U-=0.1*U.grad        
        U.grad.zero_()
        
#%%

#    indx=torch.randint(0,NF,(NF,))
#    D=vt[:,indx]
#    #u=u1.permute(3,1,2,0)
#    u=torch.reshape(u1,(nbasis,nx*nx*2))
#    #x=torch.matmul(u,D)
#    x=torch.mm(D.T,u)
#    u2=torch.mm(D,x)
#    #x=x.permute(3,0,1,2)
#    x=torch.reshape(x,(NF,nx,nx,2))
#    x=x.permute(0,3,1,2)
#    trnDs=rd.DataGen(x.cpu().numpy())
#    ldr=DataLoader(trnDs,batch_size=1,shuffle=False,num_workers=8)
#    with torch.no_grad():
#        for data in (ldr):
#            slcR=unet(data.to(gpu,dtype))
#            y.append(slcR)
#    Y=torch.cat(y)
#    Y=Y.permute(0,2,3,1)
#    Y=torch.reshape(Y,(NF,nx*nx*2))
#    z=torch.mm(D,Y)
#    #z=z.permute(3,0,1,2)
#    z=torch.reshape(z,(nbasis,nx,nx,2))
#    u2=torch.reshape(u2,(nbasis,nx,nx,2))
#    z=u2-z
#    return z  
#%%
#sf.tic()
#recT,err=sf.pt_cg(AtA1,atbT,atbT,cgIter1,cgTol)
#sf.toc()
#%%
rec = np.squeeze(U.detach().cpu().numpy())
rec=np.reshape(rec,(nbasis,nx,nx,2))
rec = rec[:,:,:,0] + 1j*rec[:,:,:,1]
#for i in range(30):
#    plt.imshow((np.squeeze(np.abs(xx[27,:,:]))),cmap='gray')       
#    plt.show() 
#    plt.pause(1)
#xx=np.fft.fftshift(np.fft.fftshift(atbV,1),2)
rec=np.reshape(rec,(nbasis,nx*nx))
rec1=rec.T@V
rec1=np.reshape(rec1,(nx,nx,NF))
rec1=np.fft.fftshift(np.fft.fftshift(rec1,0),1)
#plt.figure(1,[15,15])
for i in range(30):
    plt.imshow((np.squeeze(np.abs(rec1[:,:,i]))),cmap='gray')       
    plt.show() 
    plt.pause(0.05)

#sio.savemat('temp33_prp300.mat',{"rec":rec1}) 
    
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