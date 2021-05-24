"""
Created on Tue Apr 14 11:36:39 2020
This is the cart_UV code in pytorch
@author: abdul haseeb
"""
import os,time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import gen_net as gn
import gen_net3 as gn2
import gen_netV as gnV

import supportingFun as sf
import supporting_storm_fun as ss
from torch.autograd import Variable
import h5py as h5
import readData as rd
import scipy.io as sio
import sys
from torch.utils.data import TensorDataset
import random

sys.path.insert(0, "/Users/ahhmed/pytorch_unet")

from torchkbnufft.torchkbnufft import KbNufft
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from unetmodel import UnetClass
from dn_modelV import SmallModel

from scipy import ndimage
import gen_net1 as gn1
gpu=torch.device('cuda:0')
cpu=torch.device('cpu')
directory='/Users/ahhmed/Codes/TensorFlow_SToRM/Data1/FID142454v/d900/'
directory2='/Users/ahhmed/Codes/TensorFlow_SToRM/Data1/FID142454v/'

dtype = torch.float
directory1='/Users/ahhmed/pytorch_unet/SPTmodels/19Jun_101938pm_50ep_20Jun/'
#directory1='/Users/ahhmed/pytorch_unet/SPTmodels/21Jun_110930pm_100ep_21Jun/'
#directory1='/Users/ahhmed/pytorch_unet/SPTmodels/23Jun_111055pm_150ep_23Jun/'

#chkPoint='24'
chkPoint='52'


NF=900#100#900 
nx=512
#N=400
N1=300
N2=50
nch=4
thres=0.05
nbasis=30
lam=0.01
st=0
batch_sz=100
TF=900
#%%
G=UnetClass().to(gpu)
GV=SmallModel().to(gpu)
G.load_state_dict(torch.load('wts-2U96.pt'))
GV.load_state_dict(torch.load('wts-2V96.pt'))
#optimizer=torch.optim.SGD([{'params':G.parameters(),'lr':5e-3,'momentum':0.9}])
optimizer=torch.optim.AdamW([{'params':G.parameters(),'lr':1e-4},{'params':GV.parameters(),'lr':1e-4}])
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=6, verbose=True, min_lr=5e-5)

trnFiles=os.listdir('/Shared/lss_jcb/abdul/prashant_cardiac_data/Data/')
sz=len(trnFiles)
#data2=np.zeros((sz,1,n_select,N,N)).astype(np.complex64)
rndm=random.sample(range(sz),sz)
for fl in range(sz):
    dictn=sio.loadmat('/Shared/lss_jcb/abdul/prashant_cardiac_data/Data/'+trnFiles[rndm[fl]])
    #dictn=sio.loadmat('/Shared/lss_jcb/abdul/prashant_cardiac_data/meas_MID00185_FID142454_SAX_gNAV_gre11.mat')
    #with h5.File(directory2+'bCom_img1.h5','r') as f:
    #  # load the L matrix
    #  nav_r=np.asarray(f['/bCom_img/re'])
    #  nav_i=np.asarray(f['/bCom_img/im'])
    #  nav=nav_r+nav_i*1j
    #  nav=nav[0+st:NF+st]
    #  nav=nav.astype(np.complex64)
    #  nav=abs(nav)
      # load the L matrix
    L=np.asarray(dictn['L']) 
      #L=L[0+st:NF+st,0+st:NF+st]
    L=L.astype(np.complex64)
    U,sb,V=np.linalg.svd(L)
    sb[NF-1]=0
    V=V[TF-nbasis:TF,0+st:NF+st]
    sb=np.diag(sb[NF-nbasis:NF,])*lam
    V=V.astype(np.float32)
    #Vn=np.zeros((sz,nbasis,NF),dtype='float32')
    #Vn[fl]=V
    
      # coil sensitivity maps
    csm=np.asarray(dictn['csm'])
    csm=csm.astype(np.complex64)
    csm=np.transpose(csm,[2,1,0])
    
    csm=csm[0:nch,:,:]
      ##csm1=np.fft.fftshift(np.fft.fftshift(csm,1),2)
    csm=np.fft.fftshift(np.fft.fftshift(csm,1),2)
    #csmTrn=np.zeros((sz,nch,nx,nx),dtype='complex64')
    #csmTrn[fl]=csm
    #del csm
      
      
      # reading the RHS: AHb
    S=np.asarray(dictn['S'])
    data=np.asarray(dictn['b'])
    tmp=np.zeros((nx*nx,1),dtype='complex64')
    kdata=np.zeros((NF,nch,nx*nx),dtype='complex64')
    
    for i in range(NF):
        for j in range(nch):
            indx=np.squeeze(S[i,0])
            tmp[indx-1]=data[i,j]
            kdata[i,j]=np.squeeze(tmp)
            tmp=np.zeros((nx*nx,1),dtype='complex64')
    kdata=np.transpose(kdata,(1,0,2))        
    kdata=kdata[0:nch,0+st:NF+st,:]
    kdata=kdata.astype(np.complex64)
    kdata=np.reshape(kdata,(nch,int(NF/batch_sz)*batch_sz,nx,nx))
    #kdata=np.transpose(kdata,(0,1,3,2))
    kdata=np.fft.fftshift(np.fft.fftshift(kdata,2),3)
    #kdata=np.reshape(kdata,(nch,int(NF/batch_sz)*batch_sz,nx*nx))
    #kdata=np.reshape(kdata,(nch,int(NF/batch_sz)*batch_sz,nx,nx))
    #kdata=np.transpose(kdata,(0,1,3,2))
    kdata=np.reshape(kdata,(nch,int(NF/batch_sz)*batch_sz,nx*nx))
    
    kdata=kdata/np.max(np.abs(kdata))
    
    msk=np.zeros((NF,nx*nx),dtype='float32')
    for k in range(NF):
        indx=np.squeeze(S[k,0])
        msk[k,indx-1]=1
    
    maskTst=np.reshape(msk,(NF,nx,nx))
    #maskTst=np.transpose(maskTst,(0,2,1))
    maskTst=maskTst[0+st:NF+st]
    maskTst=np.fft.fftshift(np.fft.fftshift(maskTst,1),2)
    maskTst=np.reshape(maskTst,(NF,nx*nx))
    
    #res=np.reshape(res,(int(NF/batch_sz),batch_sz,nbasis,nx*nx))
    #%% make them pytorch tensors
    kdataT=torch.tensor(sf.c2r(kdata))
    csmT=torch.tensor(sf.c2r(csm))
    maskT=torch.tensor(maskTst)
    VT=torch.tensor(V)
    sT=torch.tensor(sb)
    #W=torch.tensor(W)
    #Nv=torch.tensor(nav)
    #%% take them to gpu
    
    csmT=csmT.to(gpu)
    maskT=maskT.to(dtype=torch.float32)
    VT=VT.to(gpu)
    #sT=sT.to(gpu)
    #Nv=Nv.to(gpu)
    #%%
    
    z=ss.ATBV(kdataT,csmT,VT)
    z=z.permute(3,0,1,2)
    z=torch.reshape(z,(1,2*nbasis,nx,nx)).to(gpu)
    
    #z = torch.randn((1,60,512,512),device=gpu, dtype=dtype)
    z = Variable(z,requires_grad=False)
    z1=torch.reshape(VT,(1,1,nbasis,NF))
    z1 = Variable(z1,requires_grad=False)
    #G.load_state_dict(torch.load('tempUfull.pt'))
    
        #%% class for genenrator based storm method   zz
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
                        
        def forward(self,x,mask,v1):
            #tmp2=torch.FloatTensor(self.nch,self.nbasis,self.nx,self.nx,2).fill_(0)
            nbas=x.shape[0]
            m_res=ss.maskV(mask.cuda(),v1)
            m_res=m_res.unsqueeze(3).repeat(1,1,1,2)
            m_res=torch.reshape(m_res,(self.NF,nbasis,nx*nx*2))
            tmp5=torch.cuda.FloatTensor(self.nch,self.NF,self.nx*self.nx*2).fill_(0)
            x=torch.reshape(x,(nbas,self.nx,self.nx,2))
            for i in range(nch):
                tmp2=sf.pt_fft2c(sf.pt_cpx_multipy(x,self.csmT[i].repeat(nbas,1,1,1)))
                tmp2=torch.reshape(tmp2,(nbas,self.NX))
                tmp2=tmp2.repeat(self.NF,1,1)*m_res
                tmp5[i]=tmp2.sum(axis=1)
                
            del tmp2,x    
            return tmp5.cpu()
    
    #%%
    
    torch.cuda.empty_cache()
    #res=maskV(maskTst,v1.detach().cpu().numpy())
    #maskT=torch.tensor(maskTst).to(dtype=torch.float32)
    
    loss=0
    #indx=torch.randint(0,NF,(NF,))
    #indx=torch.tensor(indx)
    #B=B[:,indx]
    #maskT=maskT[indx,]
    batch_sz=20
    bb_sz=int(NF/batch_sz)
    B=torch.reshape(kdataT,(nch,int(NF/batch_sz),batch_sz,nx*nx*2))
    maskT=torch.reshape(maskT,(int(NF/batch_sz),batch_sz,nx*nx))
    
    F=AUV(csmT,nbasis,batch_sz)
    #maskT=maskT.to(gpu)
    
    sf.tic()
    for ep1 in range(30):
        #inx=torch.randint(0,bb_sz,(bb_sz,))
        for bat in range(bb_sz):
            v1=GV(z1).squeeze(0).squeeze(0)
            v1=v1.permute(1,0)
            v1=torch.reshape(v1,(int(NF/batch_sz),batch_sz,nbasis))
            #mask_b=maskT[bat]

            u1=G(z)
            u1=torch.reshape(u1,(2,nbasis,nx,nx))
            u1=u1.permute(1,2,3,0)
            u1=torch.reshape(u1,(nbasis,nx*nx*2))
            b_est=F(u1,maskT[bat],v1[bat])

            #loss = criterion(out, target) + l1_regularization
            loss=(b_est-B[:,bat]).pow(2).sum()#+0.1*l1_reg#+(sT@W*u1).pow(2).sum()
            #if ep1%5==0:
            print(ep1,loss.item())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step(loss.detach())
    #    if ep1%20==0:
    #        torch.save(u1,'res_'+str(ep1)+'.pt')
    sf.toc()
    if fl>=40 or fl==20:
        wtsFname1='wts-4U'+str(fl+1)+'.pt'
        torch.save(G.state_dict(),wtsFname1) 
        wtsFname2='wts-4V'+str(fl+1)+'.pt'
        torch.save(GV.state_dict(),wtsFname2)      
#%%
#rec = np.squeeze(u1.detach().cpu().numpy())
#rec=np.reshape(rec,(nbasis,nx,nx,2))
#rec = rec[:,:,:,0] + 1j*rec[:,:,:,1]
#for i in range(30):
#    plt.imshow((np.squeeze(np.abs(rec[i,:,:]))),cmap='gray')       
#    plt.show() 
#    plt.pause(0.05)
##xx=np.fft.fftshift(np.fft.fftshift(atbV,1),2)
#rec=np.reshape(rec,(nbasis,nx*nx))
#rec1=rec.T@V[:,0:NF]
#rec1=np.reshape(rec1,(nx,nx,NF))
##rec1=np.fft.fftshift(np.fft.fftshift(rec1,0),1)
##plt.figure(1,[15,15])
#for i in range(100):
#    plt.imshow((np.squeeze(np.abs(rec1[:,:,i]))),cmap='gray')       
#    plt.show() 
#    plt.pause(0.05)
