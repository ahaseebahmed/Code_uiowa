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
from torch.utils.data import TensorDataset
from model import UnetClass
from scipy import ndimage

gpu=torch.device('cuda:0')
directory='/Users/ahhmed/Codes/TensorFlow_SToRM/Data1/FID142454v/d900/'
dtype = torch.float
directory1='/Users/ahhmed/pytorch_unet/SPTmodels/28Apr_030506pm_41ep_noise_1/'
chkPoint='41'

NF=400
NF1=400
NF2=NF-NF1 
nx=512
#N=400
N1=300
N2=50
nch=2
thres=0.05
nbasis=30
lam=0.01
#%%

    #range(len(trnFiles)):
#matfile = sio.loadmat('./../Codes/TensorFlow_SToRM/tstData/storm_900/FID142454_11.mat')
#data0 = matfile['D']
#data0=np.transpose(data0,(1,0))
#data1 = matfile['U1']
#data=np.matmul(data1,data0)
#data=np.transpose(data,(1,0))
#data=np.reshape(data,(900,512,512))
#data=np.transpose(data,[0,2,1])
#data=data[0:800,]
#data=np.fft.fftshift(np.fft.fftshift(data,1),2) 
##        data=data[:,110:380,110:380]
#data=data.astype(np.complex64)
#y1=data[range(0,800,2)]
#y2=data[range(1,800,2)]

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
  V1=V[:,0:NF1]
  V2=V[:,NF1:NF]
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
  kdata=kdata[0:nch,0:NF,:]
  #atb=np.squeeze(np.transpose(atb[0:NF,:,:],[0,2,1]))  
  #atb=np.fft.fftshift(np.fft.fftshift(atb,1),2)
#%%
tmp_sz=len(kdata[0,0])
indx1=np.random.randint(0,tmp_sz,int(tmp_sz/2))
#indx1=np.zeros((tmp_sz),dtype=int)
#indx1[tmp1]=tmp1
indx2=sorted(list(set(range(tmp_sz))-set(indx1)))
kdata1=np.zeros((nch,NF,nx*nx),dtype=complex)
kdata2=np.zeros((nch,NF,nx*nx),dtype=complex)

kdata1[:,:,indx1]=kdata[:,:,indx1]
kdata2=kdata-kdata1

tmp1=np.zeros((nx,nx))
res1=np.zeros((nx,nx,NF),dtype=complex)  
  
for i in range(NF):
    for j in range(nch):
        tmp=kdata1[j,i,:]
        tmp=np.expand_dims(tmp,axis=1)
        tmp=np.reshape(tmp,(nx,nx))
        tmp1=tmp1+np.fft.ifft2(tmp)*np.conj(csm[j])
    res1[:,:,i]=tmp1
    tmp1=np.zeros((nx,nx))    
res1=res1*nx
res1=np.transpose(res1,[2,1,0])
res1=np.fft.fftshift(np.fft.fftshift(res1,1),2)

tmp1=np.zeros((nx,nx))
res2=np.zeros((nx,nx,NF),dtype=complex)  
  
for i in range(NF):
    for j in range(nch):
        tmp=kdata2[j,i,:]
        tmp=np.expand_dims(tmp,axis=1)
        tmp=np.reshape(tmp,(nx,nx))
        tmp1=tmp1+np.fft.ifft2(tmp)*np.conj(csm[j])
    res2[:,:,i]=tmp1
    tmp1=np.zeros((nx,nx))    
res2=res2*nx
res2=np.transpose(res2,[2,1,0])
res2=np.fft.fftshift(np.fft.fftshift(res2,1),2)
#%%
########################################################
y1=torch.tensor(sf.c2r(res1))
y2=torch.tensor(sf.c2r(res2))

y1=y1.to(gpu)
y2=y2.to(gpu)

y1=y1.permute(0,3,1,2)
y2=y2.permute(0,3,1,2)

directory='/Users/ahhmed/pytorch_sense'

unet=UnetClass()
unet=unet.to(gpu)
optimizer=torch.optim.Adam(unet.parameters(),lr=1e-3)
def lossFun(pred,org):
    loss=torch.mean(torch.abs(pred-org))
    return loss

epLoss=0
for ep in range(400):
    trnDs=rd.DataGen(y1.detach().cpu().numpy(),y2.detach().cpu().numpy())
    #trnDs=TensorDataset(y1.detach(),y2.detach())
    ldr=DataLoader(trnDs,batch_size=1,shuffle=True,num_workers=8,pin_memory=True)
    for i, (inp,org) in enumerate(ldr):
        inp,org=inp.to(gpu,dtype),org.to(gpu,dtype)
        res=unet(inp)
        loss=lossFun(res,org)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epLoss+=loss.item()
    if ((ep+1)==200) :
        wtsFname=directory+'/twosamp-'+str(ep+1)+'.pt'
        torch.save(unet.state_dict(), wtsFname)



#%%
####################################################3
tmp1=np.zeros((nx,nx))
res=np.zeros((nx,nx,NF),dtype=complex)  
  
for i in range(NF):
    for j in range(nch):
        tmp=kdata[j,i,:]
        tmp=np.expand_dims(tmp,axis=1)
        tmp=np.reshape(tmp,(nx,nx))
        tmp1=tmp1+np.fft.ifft2(tmp)*np.conj(csm[j])
    res[:,:,i]=tmp1
    tmp1=np.zeros((nx,nx))    
res=res*nx
res=np.reshape(res,(nx*nx,NF))    
atbV1=V1@res[:,0:NF1].T
atbV2=V2@res[:,NF1:NF].T

atbV1=np.reshape(atbV1,(nbasis,nx,nx))
atbV1=atbV1.astype(np.complex64)
atbV1=np.transpose(atbV1,[0,2,1])

atbV2=np.reshape(atbV2,(nbasis,nx,nx))
atbV2=atbV2.astype(np.complex64)
atbV2=np.transpose(atbV2,[0,2,1])

del tmp,tmp1,res


with h5.File(directory+'S_mask.h5', 'r') as f: 
  maskTst=np.asarray(f['S'])
  maskTst=np.squeeze(np.transpose(maskTst[0:NF,:,:],[0,2,1]))


#%% make them pytorch tensors
atbT1=torch.tensor(sf.c2r(atbV1))
atbT2=torch.tensor(sf.c2r(atbV2))
csmT=torch.tensor(sf.c2r(csmTrn))
maskT=torch.tensor(np.tile(maskTst[...,np.newaxis],(1,1,1,2)))
VT=torch.tensor(V1)
sT=torch.tensor(sbasis)
W=torch.tensor(W)


#%% take them to gpu
torch.cuda.empty_cache()
atbT1=atbT1.to(gpu)
#atbT2=atbT2.to(gpu)
csmT=csmT.to(gpu)
maskT=maskT.to(gpu,dtype)
VT=VT.to(gpu)
W=W.to(gpu,dtype)
sT=sT.to(gpu)

#%% make A and At operators   
def AtAUV(x,csmT,maskT):
    atbv=torch.cuda.FloatTensor(nbasis,nx,nx,2).fill_(0)
    tmp6=torch.cuda.FloatTensor(nbasis,nx,nx,2).fill_(0)
    csmConj=sf.pt_conj(csmT)

    for i in range(nch):
        tmp1=sf.pt_cpx_multipy(x,csmT[i].repeat(nbasis,1,1,1))
        tmp2=sf.pt_fft2c(tmp1)
        del tmp1
        for k in range(NF1):
            tmp3=maskT[k].repeat(nbasis,1,1,1)*tmp2
            tmp4=VT[:,k].unsqueeze(1)
            tmp3=torch.reshape(tmp3,(nbasis,nx*nx*2))
            tmp5=tmp4.T@tmp3#torch.mm(tmp4.T,tmp3)
            tmp5=tmp4@tmp5#torch.mm(tmp4,tmp5)        
            tmp5=torch.reshape(tmp5,(nbasis,nx,nx,2))
            tmp6=tmp6+tmp5
        del tmp2,tmp3,tmp4,tmp5   
        tmp1=sf.pt_ifft2c(tmp6)
        tmp2=sf.pt_cpx_multipy(csmConj[i].repeat(nbasis,1,1,1),tmp1)
        atbv=atbv+tmp2
        tmp6=tmp6.fill_(0)
        del tmp1,tmp2
    x=torch.reshape(x,(nbasis,nx*nx*2))
    x=W*x
    reg=torch.mm(sT,x)
    reg=torch.reshape(reg,(nbasis,nx,nx,2))
    atbv=atbv+reg
    return atbv

#%% creating the training model


#%% creating validation model
ckpt=400
unet=UnetClass()
modelDir='wts-'+str(ckpt)+'.pt'
unet.load_state_dict(torch.load(modelDir))
unet.eval()
unet.to(gpu)
#%%
def rhs(u1,D):
    y=[]
    #u=u1.permute(3,1,2,0)
    #x=torch.matmul(u,D)
    #x=x.permute(3,0,1,2)
    u=torch.reshape(u1,(nbasis,nx*nx*2))
    x=torch.mm(D.T,u)
    x=torch.reshape(x,(NF,nx,nx,2))
    x=x.permute(0,3,1,2)
    xx=x.cpu().numpy()
    xx=np.fft.fftshift(np.fft.fftshift(xx,2),3)
    #trnDs=rd.DataGen(xx)
    trnDs=rd.DataGen(xx,xx)
    ldr=DataLoader(trnDs,batch_size=1,shuffle=False,num_workers=8)
    with torch.no_grad():
        for i, (inp,org) in enumerate(ldr):
            inp,org=inp.to(gpu,dtype),org.to(gpu,dtype)
            slcR=unet(inp)
            y.append(slcR)
    Y=torch.cat(y)
    xx=Y.detach().cpu().numpy()
    yy=np.fft.fftshift(np.fft.fftshift(xx,2),3)
    Y=torch.tensor(yy)
    Y=Y.to(gpu,dtype)
#    Y=Y.permute(2,3,1,0)
#    z=torch.matmul(Y,D.T)
#    z=z.permute(3,0,1,2)
    Y=Y.permute(0,2,3,1)
    Y=torch.reshape(Y,(NF,nx*nx*2))
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
lam2=1e-1
cgIter=10
cgTol=1e-15
out_iter=5
cgIter1=10
epLoss=0
u1=atbT1.permute(0,3,1,2)
u2=atbT2.permute(0,3,1,2)
recT=torch.zeros_like(atbT1)
AtA=lambda x: AtAUV(x,csmT,maskT)
recT=sf.pt_cg(AtA,atbT1,recT,40,cgTol)
#%% run method

vt=VT
AtA=lambda x: AtAUV(x,csmT,maskT)+lam2*reg_term(x,vt)      
    #sf.tic()
for i in range(out_iter):
    atbT=atbT1+lam2*rhs(recT,vt)
    recT=sf.pt_cg(AtA,atbT,recT,cgIter,cgTol)
    


#%%
#sf.tic()
#recT,err=sf.pt_cg(AtA1,atbT,atbT,cgIter1,cgTol)
#sf.toc()
#%%
rec = np.squeeze(aa.cpu().numpy())
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
for i in range(50):
    plt.imshow((np.squeeze(np.abs(rec1[:,:,i]))),cmap='gray')       
    plt.show() 
    plt.pause(0.05)

#sio.savemat('temp1_prp400.mat',{"rec":rec1}) 
    
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
for i in range(100):
    plt.imshow((np.squeeze(np.abs(data[i,:,:]))),cmap='gray')       
    plt.show() 
    plt.pause(0.1)