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

from unetmodel2 import UnetClass
from dn_modelV2 import SmallModel
from dn_modelU2 import SmallModel1

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
        nbas=x.shape[0]
        tmp5=torch.cuda.FloatTensor(self.nch,self.NF,self.nx*self.nx*2).fill_(0)
        x=torch.reshape(x,(nbas,self.nx,self.nx,2))
        for i in range(nch):
            tmp2=sf.pt_fft2c(sf.pt_cpx_multipy(x,self.csmT[i].repeat(nbas,1,1,1)))
            tmp2=torch.reshape(tmp2,(nbas,self.NX))
            tmp2=tmp2.repeat(self.NF,1,1)*mask
            tmp5[i]=tmp2.sum(axis=1)
            
        del tmp2,x    
        return tmp5.cpu()
#%%
#G=UnetClass().to(gpu)
G=SmallModel1().to(gpu)

GV=SmallModel().to(gpu)
#G.load_state_dict(torch.load('wts-10U1.pt'))
#GV.load_state_dict(torch.load('wts-10V5.pt'))
#GV.load_state_dict(torch.load('./PTmodels/27Oct_112451am_500ep_27oct/wts-500.pt'))

#optimizer=torch.optim.SGD([{'params':G.parameters(),'lr':5e-3,'momentum':0.9}])
optimizer=torch.optim.AdamW([{'params':G.parameters(),'lr':1e-4},{'params':GV.parameters(),'lr':1e-4}])
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=6, verbose=True, min_lr=5e-5)

trnFiles=os.listdir('/Shared/lss_jcb/abdul/prashant_cardiac_data/Data/d1/')
sz=len(trnFiles)
#data2=np.zeros((sz,1,n_select,N,N)).astype(np.complex64)

rndm=random.sample(range(sz),sz)
for ep1 in range(10):
    rndm=random.sample(range(sz),sz)
    for fl in range(sz):
        dictn=sio.loadmat('/Shared/lss_jcb/abdul/prashant_cardiac_data/Data/d1/'+trnFiles[rndm[fl]])
    
        L=np.asarray(dictn['L']) 
        L=L.astype(np.complex64)
        U,sb,V=np.linalg.svd(L)
        sb[NF-1]=0
        V=V[TF-nbasis:TF,0+st:NF+st]
        sb=np.diag(sb[NF-nbasis:NF,])*lam
        V=V.astype(np.float32)
        
          # coil sensitivity maps
        csm=np.asarray(dictn['csm'])
        csm=csm.astype(np.complex64)
        csm=np.transpose(csm,[2,1,0])
        
        csm=csm[0:nch,:,:]
          ##csm1=np.fft.fftshift(np.fft.fftshift(csm,1),2)
        csm=np.fft.fftshift(np.fft.fftshift(csm,1),2)
        #csmTrn=np.zeros((sz,nch,nx,nx),dtype='complex64')
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
        #kdata1=np.reshape(kdata1,(nch,int(NF/batch_sz)*batch_sz,nx*nx))

        kdata=kdata/np.max(np.abs(kdata))
#        kdata1=kdata1/np.max(np.abs(kdata1))
#        
#        tmp1=np.zeros((nx,nx))
#        atbV=np.zeros((nx*nx,nbasis),dtype=complex)  
#        for i in range(NF):
#            for j in range(nch):
#                tmp=kdata[j,i,:]
#                tmp=np.expand_dims(tmp,axis=1)
#                tmp=np.reshape(tmp,(nx,nx))
#                tmp1=tmp1+np.fft.ifft2(tmp)*np.conj(csm[j])
#                
#            tmp2=np.expand_dims(V[:,i],axis=1)
#            tmp3=np.reshape(tmp1,(nx*nx,1))
#            atbV=atbV+np.matmul(tmp3,tmp2.T)
#            tmp1=np.zeros((nx,nx))
#        
#        atbV=atbV*nx
#        atbV=np.transpose(atbV,[1,0])
#        atbV=np.reshape(atbV,(nbasis,nx,nx))
#        atbV=atbV.astype(np.complex64)
#        atbV=np.fft.fftshift(np.fft.fftshift(atbV,1),2)
#        #atbV=np.transpose(atbV,[0,2,1])
#        del tmp,tmp1,tmp2,tmp3
        
        
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
        #atbT=torch.tensor(sf.c2r(atbV))

        csmT=torch.tensor(sf.c2r(csm))
        maskT=torch.tensor(maskTst)
        VT=torch.tensor(V)
        #%% take them to gpu    
        csmT=csmT.to(gpu)
        maskT=maskT.to(gpu,dtype=torch.float32)
        VT=VT.to(gpu)
        #%%
        
        z=ss.ATBV(kdataT,csmT,VT)
        
        maskT=torch.reshape(maskT,(NF,nx,nx))
        AtA1=lambda x: ss.AtAUV(x,csmT,maskT,VT)
        recT=torch.zeros_like(z)
        sf.tic()
        recT=sf.pt_cg(AtA1,z,recT.cuda(),15,1-15)
        sf.toc()

        #del recT
        #z = torch.randn((1,60,512,512),device=gpu, dtype=dtype)
        #z = Variable(z,requires_grad=False)
        #z1 = Variable(z1,requires_grad=False)
        z=recT+torch.normal(0,0.005,(recT.shape[0],recT.shape[1],recT.shape[2],recT.shape[3])).cuda()
        z=z.permute(3,0,1,2)
        z=torch.reshape(z,(1,2*nbasis,nx,nx)).to(gpu)
        for it in range(500):
            
            u1=G(z.cuda())
            u1=torch.reshape(u1,(2,nbasis,nx,nx))
            u1=u1.permute(1,2,3,0)
            u1=torch.reshape(u1,(nbasis,nx,nx,2))
            loss=abs(u1-recT).pow(2).sum()
            print(it,loss.item())
            #torch.cuda.empty_cache()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step(loss.detach())
        
        
        sf.tic()
        for ep1 in range(500):
            for bat in range(1):
                V1=VT.cuda()+torch.normal(0,0.001,(VT.shape[0],VT.shape[1])).cuda()
                z1=torch.reshape(V1,(1,1,nbasis,NF))
                v1=GV(z1)
                #v1=v1.permute(1,0)
                loss=abs(v1[0,0]-VT.cuda()).pow(2).sum()
                if ep1%10==0:
                    print(ep1,loss.item())
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step(loss.detach())
        sf.toc()  
        
        z1=torch.reshape(VT,(1,1,nbasis,NF))
        maskT=torch.reshape(maskT,(NF,nx*nx))

        #G.load_state_dict(torch.load('tempUfull.pt'))
        
        #%%    
        torch.cuda.empty_cache()
        
        loss=0
        batch_sz=25
        bb_sz=int(NF/batch_sz)
        B=torch.reshape(kdataT,(nch,int(NF/batch_sz),batch_sz,nx*nx*2))
        maskT=torch.reshape(maskT,(int(NF/batch_sz),batch_sz,nx*nx))
        
        F=AUV(csmT,nbasis,batch_sz)
        maskT=maskT.to(gpu)
        
        sf.tic()

            #inx=torch.randint(0,bb_sz,(bb_sz,))
        for it in range(50):    
            for bat in range(bb_sz):
                v1=(z1).squeeze(0).squeeze(0)
                v1=v1.permute(1,0)
                v1=torch.reshape(v1,(int(NF/batch_sz),batch_sz,nbasis))
                #mask_b=maskT[bat]
                m_res=ss.maskV(maskT[bat],v1[bat])
                m_res=m_res.unsqueeze(3).repeat(1,1,1,2)
                m_res=torch.reshape(m_res,(batch_sz,nbasis,nx*nx*2))
                u1=G(z)
                u1=torch.reshape(u1,(2,nbasis,nx,nx))
                u1=u1.permute(1,2,3,0)
                u1=torch.reshape(u1,(nbasis,nx*nx*2))
                b_est=F(u1,m_res.to(gpu))
    #            l1_reg = 0.
    #            for param in G.parameters():
    #                l1_reg += param.abs().sum()
                #loss = criterion(out, target) + l1_regularization
                loss=(b_est-B[:,bat]).pow(2).sum()#+0.1*l1_reg#+(sT@W*u1).pow(2).sum()
                loss=loss.cuda()
                #if ep1%5==0:
                print(ep1,fl,it,loss.item())
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step(loss.detach())
                torch.cuda.empty_cache()
    #    if ep1%20==0:
    #        torch.save(u1,'res_'+str(ep1)+'.pt')
        sf.toc()
#    if ep1%2==0:
#        wtsFname1='wts-10U'+str(ep1+1)+'.pt'
#        torch.save(G.state_dict(),wtsFname1) 
#        wtsFname2='wts-10V'+str(ep1+1)+'.pt'
#        torch.save(GV.state_dict(),wtsFname2)      
#%%
#V=v1.detach().cpu().numpy()
##
#rec = np.squeeze(u1.detach().cpu().numpy())
#rec=np.reshape(rec,(nbasis,nx,nx,2))
#rec = rec[:,:,:,0] + 1j*rec[:,:,:,1]
#for i in range(30):
#    plt.imshow((np.squeeze(np.abs(rec[i,:,:]))),cmap='gray')       
#    plt.show() 
#    plt.pause(0.05)
##    
#rec=np.reshape(rec,(nbasis,nx*nx))
#rec1=rec.T@V[:,0:NF]
#rec1=np.reshape(rec1,(nx,nx,NF))
##rec1=np.fft.fftshift(np.fft.fftshift(rec1,0),1)
##plt.figure(1,[15,15])
#for i in range(100):
#    plt.imshow((np.squeeze(np.abs(rec1[100:400,100:400,i]))),cmap='gray')       
#    plt.show() 
#    plt.pause(0.05)
##    plt.pause(0.05)
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
