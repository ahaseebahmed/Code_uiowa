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
from scipy import ndimage

gpu=torch.device('cuda:0')
directory='/Users/ahhmed/Codes/TensorFlow_SToRM/Data1/FID142454v/d900/'
dtype = torch.float
directory1='/Users/ahhmed/pytorch_unet/SPTmodels/12May_043326am_400ep_12May/'
chkPoint='250'

NF=400 
nx=512
#N=400
N1=300
N2=50
nch=5
thres=0.05
nbasis=30
lam=0.0
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
  csmTrn=np.transpose(csm,[0,2,1])
  #csm=np.fft.fftshift(np.fft.fftshift(csm,1),2)
  csmTrn=csmTrn[0:nch,:,:]
  ncoils=csmTrn.shape[0]
  del csm
  
with h5.File(directory+'rhs.h5', 'r') as f:  
  # reading the RHS: AHb
  atb=np.asarray(f['/rhs/re'])
  atb=atb+np.asarray(f['/rhs/im'])*1j
  atb=atb.astype(np.complex64)/nx
  atb=np.squeeze(np.transpose(atb[0:NF,:,:],[0,2,1]))  
  #atb=np.fft.fftshift(np.fft.fftshift(atb,1),2)

atb=np.reshape(atb,(NF,nx*nx))
atbV=np.zeros((nx*nx,nbasis),dtype=complex)  
for i in range(NF):
    tmp=atb[i,:]
    tmp=np.expand_dims(tmp,axis=0)
    tmp1=V[:,i]
    tmp1=np.expand_dims(tmp1,axis=1)
    atbV=atbV+np.matmul(tmp.T,tmp1.T)
atbV=atbV*nx
atbV=np.transpose(atbV,[1,0])
atbV=np.reshape(atbV,(nbasis,nx,nx))
atbV=atbV.astype(np.complex64)
del tmp,tmp1

with h5.File(directory+'S_mask.h5', 'r') as f: 
  maskTst=np.asarray(f['S'])
  maskTst=np.squeeze(np.transpose(maskTst[0:NF,:,:],[0,2,1]))


#%% make them pytorch tensors
atbT=torch.tensor(sf.c2r(atbV))
csmT=torch.tensor(sf.c2r(csmTrn))
maskT=torch.tensor(np.tile(maskTst[...,np.newaxis],(1,1,1,2)))
VT=torch.tensor(V)
sT=torch.tensor(sbasis)
W=torch.tensor(W)
#%% take them to gpu
atbT=atbT.to(gpu)
csmT=csmT.to(gpu)
maskT=maskT.to(gpu,dtype)
VT=VT.to(gpu)
W=W.to(gpu,dtype)
sT=sT.to(gpu)
#%%
unet=UnetClass()
modelDir=directory1+'wts-'+str(chkPoint)+'.pt'
unet.load_state_dict(torch.load(modelDir))
unet.eval()
unet.to(gpu)
#%% make A and At operators   
def AtAUV(x,csmT,maskT):
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
            tmp3=maskT[k].repeat(nbasis,1,1,1)*tmp2
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
    return atbv
#%%
#def rhs(u1,D):
#    y=[]
#    u=u1.permute(3,1,2,0)
#    x=torch.matmul(u,D)
#    x=x.permute(3,0,1,2)
#    trnDs=rd.DataGen(x.cpu().numpy())
#    ldr=DataLoader(trnDs,batch_size=1,shuffle=False,num_workers=8)
#    with torch.no_grad():
#        for data in (ldr):
#            slcR=unet(data.to(gpu,dtype))
#            y.append(slcR)
#    Y=torch.cat(y)
#    Y=Y.permute(2,3,1,0)
#    z=torch.matmul(Y,D.T)
#    z=z.permute(3,0,1,2)
#    return z 
#%% create model and load the weights
#def reg_term(u1,D):
##    y=[]
##    indx=torch.randint(0,NF,(50,))
##    D=vt[:,indx]
#    u=u1.permute(3,1,2,0)
#    x=torch.matmul(u,D)
#    u2=torch.matmul(x,D.T)
##    x=x.permute(3,0,1,2)
##    trnDs=rd.DataGen(x.cpu().numpy())
##    ldr=DataLoader(trnDs,batch_size=1,shuffle=False,num_workers=8)
##    with torch.no_grad():
##        for data in (ldr):
##            slcR=unet(data.to(gpu,dtype))
##            y.append(slcR)
##    Y=torch.cat(y)
##    Y=Y.permute(2,3,1,0)
##    z=torch.matmul(Y,D.T)
##    z=z.permute(3,0,1,2)
#    u2=u2.permute(3,1,2,0)
##    z=u2-z
#    return u2 
#%%
def reg_term1(u1,vt,NF,lam2):
    y=[]
    indx=torch.randint(0,NF,(NF,))
    D=vt[:,indx]
    u=u1.permute(3,1,2,0)
    x=torch.matmul(u,D)
    u2=torch.matmul(x,D.T)
    x=x.permute(3,0,1,2)
    xx=x.cpu().numpy()
    xx=np.fft.fftshift(np.fft.fftshift(xx,2),3)
    trnDs=rd.DataGen(xx)
    ldr=DataLoader(trnDs,batch_size=1,shuffle=False,num_workers=8)
    with torch.no_grad():
        for data in (ldr):
            slcR=unet(data.to(gpu,dtype))
            y.append(slcR)
    Y=torch.cat(y)
    xx=Y.detach().cpu().numpy()
    xx=np.fft.fftshift(np.fft.fftshift(xx,2),3)
    Y=torch.tensor(xx)
    Y=Y.to(gpu,dtype)
    Y=Y.permute(2,3,1,0)
    z=torch.matmul(Y,D.T)
    z=z.permute(3,0,1,2)
    u2=u2.permute(3,1,2,0)
    z=u2-z
    return z  
#%%
lam2=1e-2
cgIter=5
cgTol=1e-15
out_iter=5
cgIter1=40

AtA1=lambda x: AtAUV(x,csmT,maskT)#+lam2*reg_term1(x,VT,NF,lam2)
#%% run method
#recT=atbT
#sf.tic()
#for i in range(out_iter):
#    indx=torch.randint(0,NF,(50,))
#    vt=VT[:,indx]
#    AtA=lambda x: AtAUV(x,csmT,maskT)+lam2*reg_term(x,vt)      
#    atbT1=atbT+lam2*rhs(recT,vt)
#    recT=sf.pt_cg(AtA,atbT1,cgIter,cgTol)
#sf.toc()

#%%
recT=torch.zeros_like(atbT)
sf.tic()
recT=sf.pt_cg(AtA1,atbT,recT,cgIter1,cgTol)
sf.toc()
#%%
rec = np.squeeze(recT.cpu().numpy())
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
for i in range(10):
    plt.imshow((np.squeeze(np.abs(rec1[:,:,i]))),cmap='gray')       
    plt.show() 
    plt.pause(0.05)

#sio.savemat('temp_stm900.mat',{"rec":rec1}) 
    
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