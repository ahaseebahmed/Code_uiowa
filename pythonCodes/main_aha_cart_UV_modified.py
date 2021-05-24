"""
Created on Tue Apr 14 11:36:39 2020
This is the cart_UV code in pytorch
@author: abdul haseeb
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import gen_net as gn
import gen_net2 as gn2

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
from torch.optim.lr_scheduler import ReduceLROnPlateau

#from model import UnetClass
from spatial_model6 import SmallModel

from scipy import ndimage
import gen_net1 as gn1
gpu=torch.device('cuda:0')
cpu=torch.device('cpu')
directory='/Users/ahhmed/Codes/TensorFlow_SToRM/Data1/FID142454v/d900/'
#directory='/Users/ahhmed/Codes/TensorFlow_SToRM/Data1/FID142454_12/'

dtype = torch.float
directory1='/Users/ahhmed/pytorch_unet/SPTmodels/19Jun_101938pm_50ep_20Jun/'
#directory1='/Users/ahhmed/pytorch_unet/SPTmodels/21Jun_110930pm_100ep_21Jun/'
#directory1='/Users/ahhmed/pytorch_unet/SPTmodels/23Jun_111055pm_150ep_23Jun/'

#chkPoint='24'
chkPoint='52'


NF=100#100#900 
nx=512
#N=400
N1=300
N2=50
nch=4
thres=0.05
nbasis=30
lam=0.001
st=0
batch_sz=100
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
  cimg=np.transpose(cimg,[0,1,2])
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
kdata=np.reshape(kdata,(nch,int(NF/batch_sz)*batch_sz,nx*nx))
kdata=kdata/np.max(np.abs(kdata))
  #atb=np.squeeze(np.transpose(atb[0:NF,:,:],[0,2,1]))  
  #atb=np.fft.fftshift(np.fft.fftshift(atb,1),2)

#tmp1=np.zeros((nx,nx))
#atbV=np.zeros((nx*nx,nbasis),dtype=complex)  
#for i in range(NF):
#    for j in range(nch):
#        tmp=kdata[j,i,:]
#        tmp=np.expand_dims(tmp,axis=1)
#        tmp=np.reshape(tmp,(nx,nx))
#        tmp1=tmp1+np.fft.ifft2(tmp)*np.conj(csm[j])
#        
#    tmp2=np.expand_dims(V[:,i],axis=1)
#    tmp3=np.reshape(tmp1,(nx*nx,1))
#    atbV=atbV+np.matmul(tmp3,tmp2.T)
#    tmp1=np.zeros((nx,nx))
#
#atbV=atbV*nx
#atbV=np.transpose(atbV,[1,0])
#atbV=np.reshape(atbV,(nbasis,nx,nx))
#atbV=atbV.astype(np.complex64)
##atbV=np.fft.fftshift(np.fft.fftshift(atbV,1),2)
#atbV=np.transpose(atbV,[0,2,1])
#del tmp,tmp1,tmp2,tmp3

#############################################################
#kdata=np.zeros((4,400,nx,nx),dtype='float32')+np.zeros((4,400,nx,nx),dtype='float32')*1j
#with h5.File(directory+'rhs.h5', 'r') as f:  
#  # reading the RHS: AHb
#  atb=np.asarray(f['/rhs/re'])
#  atb=atb+np.asarray(f['/rhs/im'])*1j
#  atb=atb.astype(np.complex64)
#
#for fr in range(400):
#    for c in range(4):
#        kdata[c,fr]=np.fft.fft2(atb[fr]*csm[c])
#
#kdata=np.reshape(kdata,(4,400,nx*nx))
               
############################################################
with h5.File(directory+'S_mask.h5', 'r') as f: 
  maskTst=np.asarray(f['S'])
  #maskTst=np.fft.fftshift(np.fft.fftshift(maskTst,1),2)
  maskTst=np.squeeze(np.transpose(maskTst[0+st:NF+st,:,:],[0,1,2]))
  #maskTst=np.tile(maskTst[...,np.newaxis],(1,1,1,2))
  maskTst=np.reshape(maskTst,(NF,nx*nx))

res=np.zeros((NF,nbasis,nx*nx))
for k in range(NF):
    tmp3=np.tile(maskTst[k],(nbasis,1))#*tmp2
    res[k]=np.diag(V[:,k])@tmp3

res=np.reshape(res,(int(NF/batch_sz),batch_sz,nbasis,nx*nx))
#%% make them pytorch tensors
B=torch.tensor(sf.c2r(kdata))
#atbT=torch.tensor(sf.c2r(atbV))
csmT=torch.tensor(sf.c2r(csmTrn))
maskT=torch.tensor(res).to(dtype=torch.float16)
VT=torch.tensor(V)
sT=torch.tensor(sbasis)
W=torch.tensor(W)
#%% take them to gpu
B=B.to(gpu)
#atbT=atbT.to(gpu)
csmT=csmT.to(gpu)
#maskT=maskT.to(gpu,dtype=torch.float16)
VT=VT.to(gpu)
W=W.to(gpu,dtype)
sT=sT.to(gpu)

#%% make A and At operators   
class AUV(nn.Module):
    def __init__(self,csmT,nbasis,bat_sz):
        super(AUV,self).__init__()
        self.csmT=csmT
        #self.mask=maskT
        self.nbasis=nbasis
        self.NF=bat_sz
        self.nch=csmT.shape[0]
        self.nx=csmT.shape[1]
        self.NX=self.nx*self.nx*2
        #self.tmp2=torch.FloatTensor(self.nch,self.nbasis,self.nx,self.nx,2).fill_(0)
        #self.tmp5=torch.FloatTensor(self.NF,self.nch*self.nx*self.nx*2).fill_(0)
                    
    def forward(self,x,mask):
        #tmp2=torch.FloatTensor(self.nch,self.nbasis,self.nx,self.nx,2).fill_(0)
        tmp5=torch.cuda.FloatTensor(self.nch,self.NF,self.nx*self.nx*2).fill_(0)
        x=torch.reshape(x,(self.nbasis,self.nx,self.nx,2))
        for i in range(nch):
            tmp2=sf.pt_fft2c(sf.pt_cpx_multipy(x,self.csmT[i].repeat(self.nbasis,1,1,1)))
            tmp2=torch.reshape(tmp2,(self.nbasis,self.NX))
            tmp2=tmp2.repeat(self.NF,1,1)*mask
            tmp5[i]=tmp2.sum(axis=1)

        return tmp5

#%%
#res=torch.cuda.FloatTensor(NF,nbasis,nx*nx*2).fill_(0)
#for k in range(NF):
#    tmp3=maskT[k].repeat(nbasis,1)#*tmp2
#    res[k]=torch.diag_embed(VT[:,k])@tmp3
    
#B=B.permute(1,0,2,3)
loss=0
bb_sz=int(NF/batch_sz)
B=torch.reshape(B,(nch,int(NF/batch_sz),batch_sz,nx*nx*2))
maskT=torch.reshape(maskT,(int(NF/batch_sz),batch_sz,nbasis,nx*nx))

G=gn2.generator().to(gpu)
G.load_state_dict(torch.load('tempUfull.pt'))
#G.eval()
#G.weight_init(mean=0.0, std=0.02)

F=AUV(csmT,nbasis,batch_sz)

#U = torch.zeros((nbasis,nx*nx*2),requires_grad=True,device=gpu, dtype=dtype)
#U=atbT
#U=Variable(U,requires_grad=True)

z = torch.randn((1,10,32,32),device=gpu, dtype=dtype)
z = Variable(z,requires_grad=True)
#z1=z

#optimizer=torch.optim.SGD([{'params':z,'lr':1e-2,'momentum':0.9},{'params':G.parameters(),'lr':1e-2,'momentum':0.9}])
#optimizer=torch.optim.SGD([{'params':G.parameters(),'lr':5e-3,'momentum':0.9}])
#optimizer=torch.optim.SGD([{'params':U,'lr':5e-2,'momentum':0.9}])
optimizer=torch.optim.Adam([{'params':z,'lr':1e-4},{'params':G.parameters(),'lr':1e-4}])
#optimizer=torch.optim.Adam([{'params':U,'lr':1e-2}])

scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=6, verbose=True, min_lr=5e-5)

#for ep1 in range(10):
    
for bas in range(29,-1,-1):
    for ep2 in range(1000):   
        for bat in range(bb_sz):
            mask_b=maskT[bat].cuda()
#            maskb=mask_b[:,nbasis-1].unsqueeze(1)
#            if bas!=nbasis-1:                
#                for t1 in range(bas,nbasis):                
#                    maskb=torch.cat((maskb,mask_b[:,t1]),dim=1)
            
            mask_b=mask_b.unsqueeze(3).repeat(1,1,1,2)
            mask_b=torch.reshape(mask_b,(batch_sz,nbasis,nx*nx*2))
            U=G(z)
            ##U=U.permute(1,2,3,0)
            U=torch.reshape(U,(2,nbasis,nx,nx))
            U=U.permute(1,2,3,0)
            U=torch.reshape(U,(nbasis,nx*nx*2))
#            for t2 in range(bas):                
#                U[t2,:]=torch.zeros((nx*nx*2))
            b_est=F(U,mask_b)
            loss=0.5*(b_est-B[:,bat]).pow(2).sum()#+(sT@W*U).pow(2).sum()
            if ep2%5==0:
                print(ep2,loss.item())
            
            optimizer.zero_grad()
            #G.zero_grad()
            loss.backward()
            optimizer.step()
            #scheduler.step(loss.detach())
            #z1=z
            
#        with torch.no_grad():
#            z-=0.1*z.grad
#            for param in G.parameters():
#                param-=0.01*param.grad
#            z.grad.zero_()
#%%

#G=gn1.generator().to(gpu)
#z = torch.randn((1,10,32,32),device=gpu, dtype=dtype)
#z = Variable(z,requires_grad=True)
##optimizer=torch.optim.SGD([{'params':G.parameters(),'lr':5e-3,'momentum':0.9}])
#optimizer=torch.optim.Adam([{'params':z,'lr':1e-4},{'params':G.parameters(),'lr':1e-4}])
#scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=6, verbose=True, min_lr=5e-5)
#
##z1=z
#
#for ep1 in range(10000):
#    for bat in range(1):
#        u1=G(z)
#        #U=U.permute(1,2,3,0)
#        u1=torch.reshape(u1,(2,nbasis,nx,nx))
#        u1=u1.permute(1,2,3,0)
#        u1=torch.reshape(u1,(nbasis,nx*nx*2))
#        loss=abs(u1-U).pow(2).sum()
#        if ep1%10==0:
#            print(ep1,loss.item())
#        
#        optimizer.zero_grad()
#        loss.backward()
#        optimizer.step()
#        scheduler.step(loss.detach())
#        
#torch.save(G.state_dict(),'tempU.pt')

#%%
rec = np.squeeze(U.detach().cpu().numpy())
rec=np.reshape(rec,(nbasis,nx,nx,2))
rec = rec[:,:,:,0] + 1j*rec[:,:,:,1]
for i in range(30):
    plt.imshow((np.squeeze(np.abs(rec[i,:,:]))),cmap='gray')       
    plt.show() 
    plt.pause(0.05)
#xx=np.fft.fftshift(np.fft.fftshift(atbV,1),2)
rec=np.reshape(rec,(nbasis,nx*nx))
rec1=rec.T@V[:,0:NF]
rec1=np.reshape(rec1,(nx,nx,NF))
rec1=np.fft.fftshift(np.fft.fftshift(rec1,0),1)
#plt.figure(1,[15,15])
for i in range(100):
    plt.imshow((np.squeeze(np.abs(rec1[:,:,i]))),cmap='gray')       
    plt.show() 
    plt.pause(0.05)

#sio.savemat('temp33_prp_gen900.mat',{"rec":rec1}) 
 
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