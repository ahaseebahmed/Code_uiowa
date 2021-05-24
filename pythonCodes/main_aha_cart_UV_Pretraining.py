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
#optimizer=torch.optim.SGD([{'params':G.parameters(),'lr':5e-3,'momentum':0.9}])
optimizer=torch.optim.AdamW([{'params':G.parameters(),'lr':1e-4},{'params':GV.parameters(),'lr':1e-4}])
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=6, verbose=True, min_lr=5e-5)

trnFiles=os.listdir('/Shared/lss_jcb/abdul/prashant_cardiac_data/')
sz=len(trnFiles)
#data2=np.zeros((sz,1,n_select,N,N)).astype(np.complex64)
rndm=random.sample(range(sz),sz)
for fl in range(sz):
dictn=sio.loadmat('/Shared/lss_jcb/abdul/prashant_cardiac_data/'+trnFiles[fl])
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
#skdata1=np.zeros((sz,nch,NF,nx*nx),dtype='complex64')
#kdata1[fl]=kdata


  #atb=np.squeeze(np.transpose(atb[0:NF,:,:],[0,2,1]))  

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
#atbV=np.fft.fftshift(np.fft.fftshift(atbV,1),2)
##atbV=np.transpose(atbV,[0,2,1])
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
#maskTst=np.asarray(dictn['S'])

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
maskT=maskT.to(gpu,dtype=torch.float32)
VT=VT.to(gpu)
sT=sT.to(gpu)
#Nv=Nv.to(gpu)
#%%
def ATBV(x,csmT,VT):
    
    tmp1=torch.zeros((nx,nx,2)).to(gpu)
    atbv=torch.zeros((nx*nx*2,nbasis)).to(gpu)  
    for i in range(NF):
        for j in range(nch):
            tmp=x[j,i]
            tmp=torch.reshape(tmp,(nx,nx,2))
            tmp1=tmp1+sf.pt_cpx_multipy(sf.pt_ifft2c(tmp),sf.pt_conj(csmT[j]))
            
        tmp2=VT[:,i].unsqueeze(1)
        tmp3=torch.reshape(tmp1,(nx*nx*2,1))
        atbv=atbv+tmp3@tmp2.T
        tmp1=torch.zeros((nx,nx,2)).to(gpu)
    
    atbv=atbv*nx
    atbv=atbv.permute(1,0)
    atbv=torch.reshape(atbv,(nbasis,nx,nx,2))
    #atbV=atbV.astype(np.complex64)
    #atbV=np.fft.fftshift(np.fft.fftshift(atbV,1),2)
    #atbV=np.transpose(atbV,[0,2,1])
    del tmp,tmp1,tmp2,tmp3
    return atbv


#%%
def maskV(msk,Vv):
    bsz=msk.size(0)
    res=torch.zeros((bsz,nbasis,nx*nx))
    res=res.to(gpu)
    for k in range(bsz):
        tmp3=msk[k].repeat(nbasis,1)#*tmp2
        res[k]=torch.diag(Vv[k])@tmp3
    return res

#%%

#G.load_state_dict(torch.load('tempUfull.pt'))

z=ATBV(kdataT.to(gpu),csmT,VT)
z=z.permute(3,0,1,2)
z=torch.reshape(z,(1,2*nbasis,nx,nx)).to(gpu)

#z = torch.randn((1,60,512,512),device=gpu, dtype=dtype)
z = Variable(z,requires_grad=False)
#optimizer=torch.optim.SGD([{'params':G.parameters(),'lr':5e-3,'momentum':0.9}])
#sf.tic()
#for ep1 in range(20000):
#    for bat in range(1):
#        u1=G(z)
#        #U=U.permute(1,2,3,0)
#        u1=torch.reshape(u1,(2,nbasis,nx,nx))
#        u1=u1.permute(1,2,3,0)
#        u1=torch.reshape(u1,(nbasis,nx,nx,2))
#        loss=abs(u1-recT).pow(2).sum()
#        if ep1%10==0:
#            print(ep1,loss.item())
#        
#        optimizer.zero_grad()
#        loss.backward()
#        optimizer.step()
#        scheduler.step(loss.detach())
#sf.toc()        
#torch.save(G.state_dict(),'tempU.pt')
#del recT

#%%

#z1 = torch.randn((1,30,900),device=gpu, dtype=dtype)
z1=torch.reshape(VT,(1,1,nbasis,NF))
z1 = Variable(z1,requires_grad=False)
#G.load_state_dict(torch.load('tempUfull.pt'))


#sf.tic()
#for ep1 in range(20000):
#    for bat in range(1):
#        v1=GV(z1)
#        #v1=v1.permute(1,0)
#        loss=abs(v1[0,0]-VT).pow(2).sum()
#        if ep1%10==0:
#            print(ep1,loss.item())
#        
#        optimizer.zero_grad()
#        loss.backward()
#        optimizer.step()
#        scheduler.step(loss.detach())
#sf.toc() 
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
        return tmp5

#%%
#class ATUV(nn.Module):
#    def __init__(self,csmT,nbasis,bat_sz):
#        super(AUV,self).__init__()
#        self.csmT=csmT
#        #self.mask=maskT
#        self.nbasis=nbasis
#        self.NF=bat_sz
#        self.nch=csmT.shape[0]
#        self.nx=csmT.shape[1]
#        self.NX=self.nx*self.nx*2
#        #self.tmp2=torch.FloatTensor(self.nch,self.nbasis,self.nx,self.nx,2).fill_(0)
#        #self.tmp5=torch.FloatTensor(self.NF,self.nch*self.nx*self.nx*2).fill_(0)
#                    
#    def forward(self,x,mask,Vb):
#        #tmp2=torch.FloatTensor(self.nch,self.nbasis,self.nx,self.nx,2).fill_(0)
#        nbas=x.shape[0]
#        tmp5=torch.cuda.FloatTensor(self.nch,self.NF,self.nx*self.nx*2).fill_(0)
#        x=torch.reshape(x,(self.nch,self,self.NF,self.nx,self.nx,2))
#        for i in range(nch):
#            tmp2=sf.pt_fft2c(sf.pt_cpx_multipy(x,self.csmT[i].repeat(nbas,1,1,1)))
#            tmp2=torch.reshape(tmp2,(nbas,self.NX))
#            tmp2=tmp2.repeat(self.NF,1,1)*mask
#            tmp5[i]=tmp2.sum(axis=1)
#            
#        del tmp2,x    
#        return tmp5
#%%

torch.cuda.empty_cache()
#res=maskV(maskTst,v1.detach().cpu().numpy())
#maskT=torch.tensor(maskTst).to(dtype=torch.float32)

loss=0
#indx=torch.randint(0,NF,(NF,))
#indx=torch.tensor(indx)
#B=B[:,indx]
#maskT=maskT[indx,]
batch_sz=10
bb_sz=int(NF/batch_sz)
B=torch.reshape(kdataT,(nch,int(NF/batch_sz),batch_sz,nx*nx*2))
maskT=torch.reshape(maskT,(int(NF/batch_sz),batch_sz,nx*nx))

F=AUV(csmT,nbasis,batch_sz)
maskT=maskT.to(gpu)

sf.tic()
for ep1 in range(10):
    for bat in range(bb_sz):
        v1=GV(z1).squeeze(0).squeeze(0)
        v1=v1.permute(1,0)
        v1=torch.reshape(v1,(int(NF/batch_sz),batch_sz,nbasis))
        #mask_b=maskT[bat]
        m_res=maskV(maskT[bat],v1[bat])
        m_res=m_res.unsqueeze(3).repeat(1,1,1,2)
        m_res=torch.reshape(m_res,(batch_sz,nbasis,nx*nx*2))
        u1=G(z)
        u1=torch.reshape(u1,(2,nbasis,nx,nx))
        u1=u1.permute(1,2,3,0)
        u1=torch.reshape(u1,(nbasis,nx*nx*2))
        b_est=F(u1,m_res.to(gpu))
        l1_reg = 0.
        for param in G.parameters():
            l1_reg += param.abs().sum()
        #loss = criterion(out, target) + l1_regularization
        loss=0.5*(b_est-B[:,bat].cuda()).pow(2).sum()#+0.1*l1_reg#+(sT@W*u1).pow(2).sum()
        #if ep1%5==0:
        print(ep1,loss.item())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(loss.detach())
#    if ep1%20==0:
#        torch.save(u1,'res_'+str(ep1)+'.pt')
sf.toc()
#        
#%%
rec = np.squeeze(u1.detach().cpu().numpy())
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
#rec1=np.fft.fftshift(np.fft.fftshift(rec1,0),1)
#plt.figure(1,[15,15])
for i in range(100):
    plt.imshow((np.squeeze(np.abs(rec1[:,:,i]))),cmap='gray')       
    plt.show() 
    plt.pause(0.05)

#sio.savemat('t100_0.5.mat',{"rec":rec1}) 
 
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