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
import gen_net3 as gn2
import gen_netV as gnV

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

from unetmodel import UnetClass
from spatial_model6 import SmallModel

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


NF=700#100#900 
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
#%%
with h5.File(directory2+'bCom_img1.h5','r') as f:
  # load the L matrix
  nav_r=np.asarray(f['/bCom_img/re'])
  nav_i=np.asarray(f['/bCom_img/im'])
  nav=nav_r+nav_i*1j
  nav=nav[0+st:NF+st]
  nav=nav.astype(np.complex64)
  nav=abs(nav)


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
  ##cimg=np.fft.fftshift(np.fft.fftshift(cimg,1),2)
  cimg=np.transpose(cimg,[0,1,2])
  cimg=np.fft.fftshift(np.fft.fftshift(cimg,1),2)
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
  ##csm1=np.fft.fftshift(np.fft.fftshift(csm,1),2)
  csmTrn=np.transpose(csm,[0,1,2])
  csmTrn=np.fft.fftshift(np.fft.fftshift(csmTrn,1),2)
  
  ncoils=csmTrn.shape[0]
  
with h5.File(directory+'kdata.h5', 'r') as f:  
  # reading the RHS: AHb
  kdata=np.asarray(f['/kdata/re'])
  kdata=kdata+np.asarray(f['/kdata/im'])*1j
  kdata=kdata[0:nch,0+st:NF+st,:]
  kdata=kdata.astype(np.complex64)
kdata=np.reshape(kdata,(nch,int(NF/batch_sz)*batch_sz,nx,nx))
kdata1=np.fft.fftshift(np.fft.fftshift(kdata,2),3)
kdata=np.reshape(kdata,(nch,int(NF/batch_sz)*batch_sz,nx*nx))
kdata1=np.reshape(kdata1,(nch,int(NF/batch_sz)*batch_sz,nx*nx))

kdata=kdata/np.max(np.abs(kdata))
kdata1=kdata1/np.max(np.abs(kdata1))

  #atb=np.squeeze(np.transpose(atb[0:NF,:,:],[0,2,1]))  

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
atbV=np.fft.fftshift(np.fft.fftshift(atbV,1),2)
#atbV=np.transpose(atbV,[0,2,1])
del tmp,tmp1,tmp2,tmp3

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
  maskTst=np.fft.fftshift(np.fft.fftshift(maskTst,1),2)

  #maskTst=np.tile(maskTst[...,np.newaxis],(1,1,1,2))
  maskTst=np.reshape(maskTst,(NF,nx*nx))

res=np.zeros((NF,nbasis,nx*nx))
for k in range(NF):
    tmp3=np.tile(maskTst[k],(nbasis,1))#*tmp2
    res[k]=np.diag(V[:,k])@tmp3

#res=np.reshape(res,(int(NF/batch_sz),batch_sz,nbasis,nx*nx))
#%% make them pytorch tensors
atbT=torch.tensor(sf.c2r(atbV))
csmT=torch.tensor(sf.c2r(csmTrn))
maskT1=torch.tensor(maskTst)
VT=torch.tensor(V)
sT=torch.tensor(sbasis)
W=torch.tensor(W)
Nv=torch.tensor(nav)
#%% take them to gpu
#B=B.to(gpu)
#atbT=atbT.to(gpu)
csmT=csmT.to(gpu)
#maskT=maskT.to(dtype=torch.float16)
maskT1=maskT1.to(gpu,dtype=torch.float16)

VT=VT.to(gpu)
W=W.to(gpu,dtype)
sT=sT.to(gpu)
Nv=Nv.to(gpu)
#%%
def maskV(msk,Vv):
    bsz=msk.size(0)
    res=torch.zeros((bsz,nbasis,nx*nx))
    res=res.to(gpu)
    for k in range(bsz):
        tmp3=msk[k].repeat(nbasis,1)#*tmp2
        res[k]=torch.diag(Vv[k])@tmp3
    return res

#%% ATA operator for CG based storm method
def AtAUV(x,csmT,maskT1):
#    atbv=torch.zeros(nbasis,nx,nx,2)
#    atbv=atbv.to(gpu)
#    tmp2=torch.zeros(nbasis,nch,nx,nx,2)
#    tmp2=tmp2.to(gpu)
    #tmp6=torch.zeros(nbasis,nx,nx,2)
    #tmp6=tmp6.to(gpu)
    atbv=torch.cuda.FloatTensor(nbasis,nx,nx,2).fill_(0)
    tmp6=torch.cuda.FloatTensor(nbasis,nx,nx,2).fill_(0)
    csmConj=sf.pt_conj(csmT)

    for i in range(nch):
        #tmp=csmT[i,:,:,:]
        #tmp=tmp.repeat(nbasis,1,1,1)
        tmp1=sf.pt_cpx_multipy(x,csmT[i].repeat(nbasis,1,1,1))
        tmp2=sf.pt_fft2c(tmp1)
        del tmp1
        for k in range(NF):
            #tmp=maskT[k,:,:,:]
            #tmp=tmp.repeat(nbasis,1,1,1).to(gpu)
            tmp3=maskT1[k].unsqueeze(2).repeat(nbasis,1,1,2)*tmp2
            #tmp3=tmp3.to(gpu,dtype)
            tmp4=VT[:,k].unsqueeze(1)
            tmp3=torch.reshape(tmp3,(nbasis,nx*nx*2))
            tmp5=tmp4.T@tmp3#torch.mm(tmp4.T,tmp3)
            #tmp5=torch.matmul(tmp3.permute(1,2,3,0),tmp4)
            #tmp5=torch.matmul(tmp5,tmp4.T)
            tmp5=tmp4@tmp5#torch.mm(tmp4,tmp5)        
            tmp5=torch.reshape(tmp5,(nbasis,nx,nx,2))
            #tmp5=tmp5.permute(3,0,1,2)
            tmp6=tmp6+tmp5
        del tmp2,tmp3,tmp4,tmp5   
        tmp1=sf.pt_ifft2c(tmp6)
        #tmp=csmConj[i,:,:,:]
        #tmp=tmp.repeat(nbasis,1,1,1).to(gpu)
        tmp2=sf.pt_cpx_multipy(csmConj[i].repeat(nbasis,1,1,1),tmp1)
        atbv=atbv+tmp2
        #tmp6=torch.zeros(nbasis,nx,nx,2)
        #tmp6=tmp6.to(gpu)
        tmp6=tmp6.fill_(0)
        del tmp1,tmp2

    x=torch.reshape(x,(nbasis,nx*nx*2))
    x=W*x
    reg=torch.mm(sT,x)
    reg=torch.reshape(reg,(nbasis,nx,nx,2))
    atbv=atbv+reg
    del x, reg
    return atbv
#%%
cgTol=1e-15
cgIter1=15

maskT1=torch.reshape(maskT1,(NF,nx,nx))
AtA1=lambda x: AtAUV(x,csmT,maskT1)
recT=torch.zeros_like(atbT)
sf.tic()
recT=sf.pt_cg(AtA1,atbT.cuda(),recT.cuda(),cgIter1,cgTol)
sf.toc()


#del AtA1, maskT1, atbT

#%%
G=UnetClass().to(gpu)
#G.load_state_dict(torch.load('tempUfull.pt'))


z = torch.randn((1,60,512,512),device=gpu, dtype=dtype)
z = Variable(z,requires_grad=True)
#optimizer=torch.optim.SGD([{'params':G.parameters(),'lr':5e-3,'momentum':0.9}])
optimizer=torch.optim.AdamW([{'params':z,'lr':1e-4},{'params':G.parameters(),'lr':1e-4}])
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=6, verbose=True, min_lr=5e-5)

#z1=z
sf.tic()
for ep1 in range(20000):
    for bat in range(1):
        u1=G(z)
        #U=U.permute(1,2,3,0)
        u1=torch.reshape(u1,(2,nbasis,nx,nx))
        u1=u1.permute(1,2,3,0)
        u1=torch.reshape(u1,(nbasis,nx,nx,2))
        loss=abs(u1-recT).pow(2).sum()
        if ep1%10==0:
            print(ep1,loss.item())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(loss.detach())
sf.toc()        
#torch.save(G.state_dict(),'tempU.pt')
#del recT

#%%

GV=gnV.generatorV(Nv.size(1)).to(gpu)
#G.load_state_dict(torch.load('tempUfull.pt'))
#optimizer=torch.optim.SGD([{'params':G.parameters(),'lr':5e-3,'momentum':0.9}])
optimizer=torch.optim.AdamW([{'params':GV.parameters(),'lr':1e-4}])
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=6, verbose=True, min_lr=5e-5)

#z1=z
sf.tic()
for ep1 in range(6000):
    for bat in range(1):
        v1=GV(Nv)
        v1=v1.permute(1,0)
        loss=abs(v1-VT).pow(2).sum()
        if ep1%10==0:
            print(ep1,loss.item())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(loss.detach())
sf.toc() 
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
##res=torch.cuda.FloatTensor(NF,nbasis,nx*nx*2).fill_(0)
##for k in range(NF):
##    tmp3=maskT[k].repeat(nbasis,1)#*tmp2
##    res[k]=torch.diag_embed(VT[:,k])@tmp3
#    
##B=B.permute(1,0,2,3)
#torch.cuda.empty_cache()
#
#loss=0
#
#
#
#indx=torch.randint(0,NF,(NF,))
#indx=torch.tensor(indx)
#B=B[:,indx]
#maskT=maskT[indx,]
##NF=300
#batch_sz=60
##B=B[:,0:NF]
#
#
##maskT=maskT[0:NF]
#
#bb_sz=int(NF/batch_sz)
#B=torch.reshape(B,(nch,int(NF/batch_sz),batch_sz,nx*nx*2))
#maskT=torch.reshape(maskT,(int(NF/batch_sz),batch_sz,nbasis,nx*nx))
#recT=torch.load('stm.pt')
#recT=torch.reshape(recT,(nbasis,nx*nx*2))
##G=gn2.generator().to(gpu)
##G.load_state_dict(torch.load('tempUfull.pt'))
##G.eval()
##G.weight_init(mean=0.0, std=0.02)
#
#F=AUV(csmT,nbasis,batch_sz)
#
##U = torch.zeros((nbasis,nx*nx*2),requires_grad=True,device=gpu, dtype=dtype)
##U=atbT
##U=Variable(U,requires_grad=True)
#
##z = torch.randn((1,10,32,32),device=gpu, dtype=dtype)
##z = Variable(z,requires_grad=True)
##z1=z
#
##optimizer=torch.optim.SGD([{'params':z,'lr':1e-2,'momentum':0.9},{'params':G.parameters(),'lr':1e-2,'momentum':0.9}])
##optimizer=torch.optim.SGD([{'params':G.parameters(),'lr':5e-3,'momentum':0.9}])
##optimizer=torch.optim.SGD([{'params':U,'lr':5e-2,'momentum':0.9}])
##optimizer=torch.optim.Adam([{'params':z,'lr':1e-4},{'params':G.parameters(),'lr':1e-4}])
##optimizer=torch.optim.Adam([{'params':U,'lr':1e-2}])
#
##scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=6, verbose=True, min_lr=5e-8)
#
##B=B.to(gpu)
#maskT=maskT.to(gpu)
#
##for ep1 in range(10):
#sf.tic()
#ep_count=20  
#for bas in range(29,-1,-1):
#    ep_count=int(ep_count*1)
#    if bas<=28:
#        ep_count=11
#        
#    for ep2 in range(ep_count):
#        loss_F=torch.zeros(1)
#        for bat in range(bb_sz):
#            mask_b=maskT[bat]
#            maskb=mask_b[:,nbasis-1].unsqueeze(1)
#            if bas!=nbasis-1:                
#                for t1 in range(nbasis-2,bas-1,-1):                
#                    maskb=torch.cat((mask_b[:,t1].unsqueeze(1),maskb),dim=1)
#            del mask_b
#            maskb=maskb.unsqueeze(3).repeat(1,1,1,2)
#            maskb=torch.reshape(maskb,(batch_sz,nbasis-bas,nx*nx*2))
#            U=G(z)
#            ##U=U.permute(1,2,3,0)
#            U=torch.reshape(U,(2,nbasis,nx,nx))
#            U=U.permute(1,2,3,0)
#            U=torch.reshape(U,(nbasis,nx*nx*2))
#            u1=U[nbasis-1].unsqueeze(0)
#            if bas!=nbasis-1:                
#                for t2 in range(nbasis-2,bas-1,-1):                
#                    u1=torch.cat((U[t2].unsqueeze(0),u1),dim=0)
#
#            b_est=F(u1,maskb)
#            loss=0.5*(b_est-B[:,bat].cuda()).pow(2).sum()#+(sT@W*U).pow(2).sum()
#            loss_F=loss_F+loss.item()
##            if ep2%5==0:
##                print(ep2,loss.item())
#            
#            optimizer.zero_grad()
#            #G.zero_grad()
#            loss.backward()
#            optimizer.step()
#            scheduler.step(loss.detach())
#        loss_U=(u1[-1]-recT[-1]).pow(2).sum()
#        print(ep2,(loss_F/bb_sz).item(),loss_U.item())
#            #z1=z
#            
##        with torch.no_grad():
##            z-=0.1*z.grad
##            for param in G.parameters():
##                param-=0.01*param.grad
##            z.grad.zero_()
##for ii in range(10):
##    ep_count=int(ep_count*0.8)
##    print(ep_count)
#sf.toc()
#%%

torch.cuda.empty_cache()

B=torch.tensor(sf.c2r(kdata1))


#res=maskV(maskTst,v1.detach().cpu().numpy())
maskT=torch.tensor(maskTst).to(dtype=torch.float32)



loss=0
indx=torch.randint(0,NF,(NF,))
indx=torch.tensor(indx)
B=B[:,indx]
maskT=maskT[indx,]
batch_sz=70
bb_sz=int(NF/batch_sz)
B=torch.reshape(B,(nch,int(NF/batch_sz),batch_sz,nx*nx*2))
maskT=torch.reshape(maskT,(int(NF/batch_sz),batch_sz,nx*nx))

F=AUV(csmT,nbasis,batch_sz)
maskT=maskT.to(gpu)

sf.tic()
for ep1 in range(100):
    for bat in range(bb_sz):
        v1=GV(Nv)
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