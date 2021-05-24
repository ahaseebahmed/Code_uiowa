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
import pickle

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

from torchkbnufft.torchkbnufft import KbNufft
from torchkbnufft.torchkbnufft import AdjKbNufft, ToepNufft
from torchkbnufft.torchkbnufft import MriSenseNufft
from torchkbnufft.torchkbnufft import AdjMriSenseNufft, ToepSenseNufft
from torchkbnufft.torchkbnufft.nufft.sparse_interp_mat import precomp_sparse_mats
from torchkbnufft.torchkbnufft.nufft.toep_functions import calc_toep_kernel
from torchkbnufft.torchkbnufft.mri.dcomp_calc import calculate_radial_dcomp_pytorch
#from ..functional.kbnufft import AdjKbNufftFunction, KbNufftFunction


#from ..math import complex_mult


from espirit.espirit import espirit, espirit_proj, ifft, fft

import scipy.io as sio
from scipy.linalg import fractional_matrix_power

from unetmodel import UnetClass
from dn_modelV2 import SmallModel
from dn_modelU2 import SmallModel1
from gen_net4 import generator

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


import matplotlib.pyplot as plt
import torch
import numpy as np
from fingerprints_aha import giveFingerPrintSVD
#  Read and create operators 
#%%
gpu=torch.device('cuda:0')

params = {'name':'parameters',
     'directory':'',
     'device':gpu,
     'filename':"",
     'dtype':torch.float,     
     
     'im_size':(340,340),   #imaege size
     'nintlPerFrame':2,    # interleaves per frame
     'nFramesDesired':250,  # number of frames in the reconstruction
     'slice':6,            # slice of the series to process
     'factor':2,           # scale image by 1/factor to save compute time
     'alpha': [5e-2,0,0,0],
     'gen_base_size': 36,  # base number of filters
     'gen_reg': 0,       # regularization penalty on generator
     'siz_l':2} 


nSpirals = 4000
params['nintlPerFrame'] = 4
params['nFramesDesired'] = int(nSpirals/params['nintlPerFrame'])
params['lr_g'] = 2e-4
params['lr_z'] = 0e-2

params["nSpirals"] = nSpirals
params["IRreprate"] = 500
params["flip"] = 15
params["TR"] = 8.0


ur,fp = giveFingerPrintSVD(params,6)
ur = torch.tensor(ur,requires_grad=False)
#ur=torch.tensor(ur)
#ur = ur.unsqueeze(-1).unsqueeze(-1).to(gpu)

#%%



NF=1000#100#900 
nx=340
nkx=2496
nintl=4
#N=400
N1=300
N2=50
nch=4
thres=0.05
nbasis=32
lam=0.008
st=0
batch_sz=100
TF=1000
thres=0.4
im_size = [nx,nx]
nf1=1000

#trnFiles=os.listdir('/Shared/lss_jcb/abdul/prashant_cardiac_data/Data/d2/')
#sz=len(trnFiles)
##data2=np.zeros((sz,1,n_select,N,N)).astype(np.complex64)
#
#rndm=random.sample(range(sz),sz)


        #%%
#fl=0
#str1=trnFiles[fl].split('_kd.mat')[0]
##dictn=sio.loadmat('/Shared/lss_jcb/abdul/prashant_cardiac_data/Data/'+trnFiles[rndm[fl]])
##d1=sio.loadmat('/Shared/lss_jcb/abdul/prashant_cardiac_data/Data/d2/'+str1+'_kd.mat')
##dictn=sio.loadmat('/Shared/lss_jcb/abdul/prashant_cardiac_data/Data/d1/'+str1+'.mat')
dictn=sio.loadmat('/Shared/lss_jcb/abdul/IRdata/Spiral_cine_3T_1leave_FB_062_IR_FA15_TBW5_6_fixedSpiral_post_full_kdata.mat')
d1=sio.loadmat('/Shared/lss_jcb/abdul/IRdata/L_IR.mat')

#%%

nx_m=nx//2-1
smapT=torch.ones((1,1,2,nx,nx)).cuda()

nuf_ob = MriSenseNufft(im_size=(nx,nx),smap=smapT).to(dtype)
nuf_ob=nuf_ob.to(gpu)

adjnuf_ob = AdjMriSenseNufft(im_size=(nx,nx), smap=smapT).to(dtype)
adjnuf_ob=adjnuf_ob.to(gpu)

ktraj=np.asarray(dictn['ktraj'])
ktraj=ktraj.astype(np.complex64)
ktraj=np.transpose(ktraj,(2,1,0))
ktraj=ktraj[0:NF]
ktraj=np.reshape(ktraj,(1,NF*nintl*nkx))*2*np.pi
#dcf=np.abs(ktraj)
dcf=np.asarray(dictn['dcf'])
dcf=torch.tensor(dcf[:,0]).unsqueeze(0).unsqueeze(0)
dcf=dcf.to(dtype)
dcf=dcf.repeat(NF,nintl,1)
dcf=torch.reshape(dcf,(1,1,1,NF*nintl*nkx))
dcf=dcf.repeat(nbasis,nch,2,1)


ktraj = np.stack((np.real(ktraj), np.imag(ktraj)), axis=1)


ktraj=np.tile(ktraj,(nbasis,1,1))
ktrajT = torch.tensor(ktraj).to(dtype)


cc=torch.zeros((1,1,2,nx,nx)).cuda()
cc[0,0,:,nx_m,nx_m]=1
#cc[0,0,:,236,247]=1

tr=ktrajT[0].unsqueeze(0)
dd=nuf_ob(cc,tr.cuda())
ee1=adjnuf_ob(dd*dcf[0,0].unsqueeze(0).unsqueeze(0).cuda(),tr.cuda())
#dcf=dcf/ee1[0,0,0,nx_m,nx_m]


#dcf[:,:,0]=0
#real_mat, imag_mat = precomp_sparse_mats(ktrajT, adjnuf_ob)
#interp_mats = {
#    'real_interp_mats': real_mat,
#    'imag_interp_mats': imag_mat
#}

#dcf=dcf+0.01
ktrajT_ch=ktrajT[0].unsqueeze(0)
dcf=dcf.cpu()
#df=torch.reshape(dcf[0,0,0],(NF,nintl*nx))
#df=df.unsqueeze(1).unsqueeze(1)
#df=df.repeat(1,3,2,1)
#df2=torch.reshape(df,(NF,3,2,nintl,nx))
#df2[:,:,:,0]=0
#df2[:,:,:,1]=0
#
#df2=torch.reshape(df2,(NF,3*2*nintl*nx))

torch.cuda.empty_cache()
#%%
kdata=np.asarray(dictn['kdata'])
kdata=np.transpose(kdata,(2,3,1,0))
kdata=kdata[0:NF,0:nch]
kdata=np.reshape(kdata,(NF,nch*nintl*nkx))
#kdata=kdata/np.max(np.abs(kdata))

L=np.asarray(dictn['L']) 
L=L.astype(np.complex64)
L=L[0:NF,0:NF]
U,sb,V=np.linalg.svd(L)
#sb[NF-1]=0
V=V[TF-nbasis:TF,0+st:NF+st]

sb=np.diag(sb[NF-nbasis:NF,])*lam
sb=torch.tensor(np.diag(sb))
V=V.astype(np.float32)

#V1=np.linalg.pinv(V@V.T)@V
#V2=V.T@V1
#V1=V.T@np.linalg.pinv(V@V.T)
#V1=V1.T
Vn=np.zeros((nbasis,NF,NF),dtype='float32')  
for i in range(nbasis):
    Vn[i,:,:]=np.diag(V[i,:])
Vn=np.reshape(Vn,(nbasis*NF,NF))
temp=Vn@kdata
temp=np.reshape(temp,(nbasis,NF,nch,nintl*nkx))        
temp=np.transpose(temp,(0,2,1,3))
temp=np.reshape(temp,(nbasis,nch,NF*nintl*nkx))
kdata=np.reshape(kdata,(NF,nch,nintl*nkx))


#Tensors
#csmTrn[0].shape
kdata = np.stack((np.real(kdata), np.imag(kdata)),axis=2)
kda1 = np.stack((np.real(temp), np.imag(temp)),axis=2)

# convert to tensor, unsqueeze batch and coil dimension
kdataT = torch.tensor(kdata).to(dtype)
kdata_ch=kdataT.permute(1,2,0,3)
kdataT=torch.reshape(kdataT,(NF,nch,2,nkx*nintl))
kdataT=kdataT.permute(1,2,0,3)
kdaT = torch.tensor(kda1).to(dtype)
#kdaT=kdaT.to(gpu)       

#%      Coilimages ans csm estimation  

kdata_ch=torch.reshape(kdata_ch,(nch,2,NF*nintl*nkx))
kdata_ch=kdata_ch.unsqueeze(0)
#kdata_ch=kdata_ch.to(gpu)
#ktrajT_ch=ktrajT_ch.to(gpu)
#dcfT=dcfT.to(gpu)        
#nuf_ob = KbNufft(im_size=im_size).to(dtype)
#nuf_ob=nuf_ob.to(gpu)

adjnuf_ob=adjnuf_ob.to(gpu)

coilimages=torch.zeros((1,nch,2,nx,nx))
#        A=lambda x: nuf_ob(x,ktrajT)
#        At=lambda x:adjnuf_ob(x,ktrajT)
#        AtA=lambda x:At(A(x))
for i in range(nch):
    coilimages[:,i]=adjnuf_ob(kdata_ch[:,i].unsqueeze(1).cuda(),ktrajT_ch.cuda())
    #ini=torch.zeros_like(temp)
    #coilimages[:,i]=sf.pt_cg(AtA,temp,ini,50,1e-15)
X=coilimages.cpu().numpy()
x=X[:,:,0]+X[:,:,1]*1j
x=np.transpose(x,(2,3,0,1))
x_f = fft(x, (0, 1, 2))
csmTrn = espirit(x_f, 6, 24, 0.15, 0.9925)
csm=csmTrn[:,:,0,:,0]
csm=np.transpose(csm,(2,0,1))

smap = np.stack((np.real(csm), np.imag(csm)), axis=1)
smap=np.tile(smap,(nbasis,1,1,1,1))



#cimg=np.sum(np.abs(np.squeeze(x)),axis=2)
#cimg1=ndimage.convolve(cimg,np.ones((13,13)))
#W=cimg1/np.max(np.abs(cimg1))
#thres=0.3
#W[W<thres]=0;
#W[W>=thres]=1;
#W=W.astype(int)
#  
##W1=ndimage.morphology.binary_closing(W,structure=np.ones((3,3)))
#W1=ndimage.morphology.binary_fill_holes(W)
#W1=W1.astype(float)
#W1=ndimage.convolve(W1,np.ones((13,13)))
#W1=W1/np.max(W1)
##W1=(1-W1)
##W=np.tile(W1,(nbasis,2,1,1))
#W1=np.reshape(W1,(1,1,1,nx,nx))
#smap=W1*smap
smapT = torch.tensor(smap).to(dtype)
#smapT=smapT.to(cpu)

del coilimages
torch.cuda.empty_cache()               
#%%     
nufft_ob = MriSenseNufft(im_size=im_size,smap=smapT).to(dtype)
nufft_ob=nufft_ob.to(gpu)

adjnufft_ob = AdjMriSenseNufft(im_size=im_size, smap=smapT).to(dtype)
adjnufft_ob=adjnufft_ob.to(gpu)

atb=torch.zeros((nbasis,im_size[0],im_size[1]))

# generating AHb results
#dcf=torch.tensor(dcf).unsqueeze(0).unsqueeze(0)
#dcf=dcf.repeat(1,1,2,1)
#dcf=dcf*2
atb=adjnufft_ob(kdaT.cuda()*dcf.cuda(),ktrajT.cuda())
#atb=atb.squeeze(1)
#atb=atb*torch.tensor(W).cuda()
#mx=torch.max((atb**2).sum(dim=-3).sqrt())
#mx1=0.0073#torch.max((yy1**2).sum(dim=-3).sqrt())

v1=torch.reshape(torch.tensor(V),(1,1,nbasis,NF,1)).cuda()
smapT=smapT.cpu()
ktrajT=ktrajT.to(cpu)

def AUV(u1):
    yy=torch.cuda.FloatTensor(nch,2,NF,nkx*nintl).fill_(0)
    for i in range(nbasis):
        xx=nufft_ob(u1[i].unsqueeze(0),ktrajT[i].unsqueeze(0).cuda(),smap=smapT[i].unsqueeze(0).cuda())
        xx=torch.reshape(xx,(nch,2,NF,nkx*nintl))
        yy=yy+v1[:,:,i]*xx

    return yy

def ATUV(yy):
    yy1=torch.cuda.FloatTensor(nbasis,1,2,nx,nx).fill_(0) 
    for i in range(nbasis):                    
        tmp=v1[:,:,i]*yy
        tmp=torch.reshape(tmp,(1,nch,2,NF*nintl*nkx))
        yy1[i]=adjnufft_ob(tmp*(dcf[i]).unsqueeze(0).cuda(),ktrajT[i].unsqueeze(0).cuda(),smap=smapT[i].unsqueeze(0).cuda())
        
    return yy1

def Sbasis1(sb,x):
    sb=sb.unsqueeze(1).unsqueeze(1).unsqueeze(1).unsqueeze(1)
    yy2=sb.cuda()*x
    return yy2
    

recT=torch.zeros_like(atb)
B=lambda x: ATUV(AUV(x))+Sbasis1(sb,x)
sf.tic()
recT=sf.pt_cg(B,atb,recT,10,1e-5)
sf.toc()
        

mx=torch.max((recT**2).sum(dim=-3).sqrt())
recT=recT/mx
kdataT1=kdataT/mx
#%%        
#recT=recT/mx
#kdataT1=kdataT/mx
#
kdata_ch=torch.reshape(kdata_ch,(1,3,2,NF,nintl*nkx))
ktrajt=torch.reshape(ktrajT,(nbasis,2,NF,nintl*nkx))
dcft=torch.reshape(dcf,(nbasis,3,2,NF,nintl*nkx))
#z1=torch.cuda.FloatTensor(3,2*nbasis,nx,nx).fill_(0)
#z2=torch.cuda.FloatTensor(3,1,nbasis,nf1).fill_(0) 
#for itt in range(3):
itt=0
#L1=np.asarray(d2['L300']) 
#L1=L1.astype(np.complex64)
#        L1=L[itt*nf1:nf1*(itt+1),itt*nf1:nf1*(itt+1)]
#        _,_,X=np.linalg.svd(L1) nmmmmmmmmmm,k
#        X=X[nf1-nbasis:nf1,:]
#        X=X.astype(np.float32)
#        #X1=X.T@np.linalg.pinv(X@X.T)
#        #X1=X1.T
#        X=np.reshape(X,(nbasis,1,1,nf1,1))
v1=np.reshape(V[:,0:nf1],(nbasis,1,1,nf1,1))
#v1=torch.reshape(v1.squeeze(0),(nbasis,1,1,nf1,1))

kdaT1=kdata_ch[:,:,:,itt*nf1:nf1*(itt+1)]
ktrajt1=ktrajt[:,:,itt*nf1:nf1*(itt+1)]
dcf1=dcft[:,:,:,itt*nf1:nf1*(itt+1)]
#            z2=VT[:,itt*nf1:nf1*(itt+1)].unsqueeze(0).unsqueeze(0).cuda()
kdaT1=kdaT1.cuda()*torch.tensor(v1).cuda()
kdaT1=torch.reshape(kdaT1,(nbasis,nch,2,nf1*nintl*nkx))
dcf1=torch.reshape(dcf1,(nbasis,nch,2,nf1*nintl*nkx))
ktrajt1=torch.reshape(ktrajt1,(nbasis,2,nf1*nintl*nkx))

atb1=adjnufft_ob(kdaT1*2*dcf1.cuda(),ktrajt1.cuda())
v2=torch.reshape(torch.tensor(V[:,0:nf1]),(1,1,nbasis,nf1,1)).cuda()

def AUV1(u1,v2):
    yy=torch.cuda.FloatTensor(nch,2,nf1,nkx*nintl).fill_(0)
    for i in range(nbasis):
        xx=nufft_ob(u1[i].unsqueeze(0),ktrajt1[i].unsqueeze(0).cuda(),smap=smapT[i].unsqueeze(0).cuda())
        xx=torch.reshape(xx,(nch,2,nf1,nkx*nintl))
        yy=yy+v2[:,:,i]*xx

    return yy

def AUV2(u1,v2):
    yy=torch.cuda.FloatTensor(nch,2,nf1,nkx*nintl).fill_(0)
    xx=nufft_ob(u1,ktrajt1.cuda(),smap=smapT.cuda())
    xx=torch.reshape(xx,(nbasis,nch,2,nf1,nkx*nintl))
    for i in range(nbasis):
        yy=yy+v2[:,:,i]*xx[i]
    return yy

def ATUV1(yy):
    yy1=torch.cuda.FloatTensor(nbasis,1,2,nx,nx).fill_(0) 
    for i in range(nbasis):                    
        tmp=v2[:,:,i]*yy
        tmp=torch.reshape(tmp,(1,nch,2,nf1*nintl*nkx))
        yy1[i]=adjnufft_ob(tmp*(dcf1[i]).unsqueeze(0).cuda(),ktrajt1[i].unsqueeze(0).cuda(),smap=smapT[i].unsqueeze(0).cuda())
        
    return yy1

recT1=torch.zeros_like(atb1)
B=lambda x: ATUV1(AUV1(x,v2))
sf.tic()
recT1=sf.pt_cg(B,atb1,recT1,10,1e-10)
sf.toc()
recT1=recT1/mx

#%% 
mmx=0.047       
#        dd=sio.loadmat('stm900o.mat')
#        org=np.asarray(dd['org']) 
#        org=org.transpose(2,0,1)


def myPSNR1(org,recon):
    sqrError=abs(org-recon)**2
    N=torch.prod(torch.tensor(org.shape[-2:]))
    mse=sqrError.sum(dim=(-1,-2))
    mse=mse/N
    
    #maxval=np.max(org,axis=(-1,-2)) + 1e-15
    psnr=10*torch.log10(mmx**2/(mse+1e-15))

    return psnr
       #atb1=atb1.squeeze(1)
def myPSNR(org,recon):

    org=torch.sqrt(torch.sum(org**2,dim=-3))
    org=org/torch.max(org)
    recon=torch.sqrt(torch.sum(recon**2,dim=-3))
    #recon=recon/torch.max(recon)
    for i in range(org.size(0)):
        t1=recon[i]
        t2=org[i]
        alpha=torch.sum(t1*t2)/torch.sum(t1*t1)
        recon[i]=alpha*t1
    sqrError=abs(org-recon)**2
    mse=sqrError.sum(dim=(-1,-2))
    N=torch.prod(torch.tensor(org.shape[-2:]))
    mse=mse/N
    org1=org.view(org.size(0),-1)
    maxval=torch.max(org1)
    #mxval=maxval.values
    #mxval=mxval.unsqueeze(1)
    psnr=10*torch.log10(1**2/(mse+1e-15))
    return psnr

def Smoothness(zvec):        
    zsmoothness = zvec[:,1:]-zvec[:,:-1]
    zsmoothness = torch.sum(zsmoothness*zsmoothness,axis=1).squeeze()
    zsmoothness = torch.sum(zsmoothness,axis=0)
    return(zsmoothness)

def KLloss(zvec1):
    loss = 0
    #Nsamples = 2
    #for j in range(Nsamples):
    mn = torch.mean(zvec1,1) #nf x latentvecotsrs
    meansub = zvec1 - mn.unsqueeze(1)
    Sigma = meansub@meansub.T/zvec1.shape[1]
    tr = torch.trace(Sigma)
    if(tr> 0.0001):
        loss = loss+ 0.5*(mn@mn.T + tr - zvec1.shape[0] - torch.logdet(Sigma))
    return(loss)         
#%%        
#G=UnetClass().to(gpu)
v1=torch.reshape(torch.tensor(V[:,0:NF]),(1,1,nbasis,NF,1)).cuda()
G=SmallModel1().to(gpu)
#G.load_state_dict(torch.load('tempp_10000.pt'))
optimizer=torch.optim.Adam([{'params':G.parameters(),'lr':1e-4}])
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=6, verbose=True, min_lr=1e-5)
#z1=recT+torch.normal(0,0.02,(recT1.shape[0],recT1.shape[1],recT1.shape[2],recT1.shape[3],recT1.shape[4])).cuda()
z=recT1
z=torch.reshape(z,(1,2*nbasis,nx,nx)).to(gpu)
pp=np.array([])

sf.tic()
for rn in range(2000):
    #z=atb#+torch.normal(0,noi[it],(atb.shape[0],atb.shape[1],atb.shape[2],atb.shape[3])).cuda()
    #z=torch.reshape(z,(1,2*nbasis,nx,nx))
        #z=atb+torch.normal(0,0.005,(atb.shape[0],atb.shape[1],atb.shape[2],atb.shape[3])).cuda()
        #z=torch.reshape(z,(1,2*nbasis,nx,nx))
    u1=G(z)#+z.cuda()
    u1=torch.reshape(u1,(nbasis,1,2,nx,nx))
    l1_reg=0
    for param in G.parameters():
        l1_reg += param.abs().sum()
    loss=((abs(u1-recT1))).pow(2).sum()#+2.0*l1_reg
    if rn%1000==0:
        plt.figure(rn)
        dd=(v1[0,0,:,20].unsqueeze(1).unsqueeze(1)*u1.squeeze(1)).sum(dim=0).detach().cpu().numpy()
        plt.imshow(np.abs((dd[0,150:350,150:300]+dd[1,150:350,150:300]*1j)),cmap='gray')
        plt.show()
        plt.pause(1)
    
    dd=(v1[0,0,:,20:30].unsqueeze(-1).unsqueeze(-1).detach().cpu()*u1[:,:,:,150:350,150:300].detach().cpu()).sum(dim=0).detach().cpu()
    org1=(v1[0,0,:,20:30].unsqueeze(-1).unsqueeze(-1).detach().cpu()*recT[:,:,:,150:350,150:300].detach().cpu()).sum(dim=0)
    psnr=torch.mean(myPSNR(org1,dd))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step(loss.detach())
    print(rn,loss.item(),psnr.item())
    pp=np.append(pp,psnr)
    
sf.toc()
ur=ur.T
ur=ur[:,50:nf1-10]
#%%
#        sf.tic()
z11=torch.normal(0,0.01,(1,1,2,1000)).cuda()
#z11=0.1*torch.ones((1,1,22,nf1)).to(gpu)
#z11[0,0,1]=-0.1
#z11[0,0,1,:]=0.01*torch.sin(torch.tensor(0.25*np.asarray(range(1000))))
#z11[0,0,0,:]=0.01*torch.sin(torch.tensor(0.05*np.asarray(range(1000))))
#z11=torch.reshape(torch.tensor(V[0:22,0:1000]),(1,1,22,1000)).to(gpu)
#z11[0,0,2]=-0.03
#z11[0,0,3]=-0.05

#z11[0,0,1,:]=torch.tensor(V[28,0:nf1])
#z11[0,0,0,:]=torch.tensor(V[26,0:nf1])
z11=z11[:,:,:,50:nf1-10]
z11 = Variable(z11,requires_grad=True)


#z1[0,0,0]=z11[0,0,0]
#z1[0,0,1]=z11[0,0,1]
#z1[0,0,2:]=ur
#
z1=torch.cat((z11,ur.unsqueeze(0).unsqueeze(0).cuda()),2)



#vv=V[30:32]
#z1=torch.reshape(torch.tensor(V[29:31,0:nf1]),(1,1,2,nf1)).to(gpu)
#z1 = Variable(z1,requires_grad=True)
V1=torch.tensor(V[6:32:,50:nf1-10]).cuda()#+torch.normal(0,0.01,(V.shape[0],V.shape[1])).cuda()
#V1=torch.reshape(V1,(1,1,nbasis,nf1))

GV=SmallModel(32).to(gpu)
#GV.load_state_dict(torch.load('temppVIR.pt')) 
#GV=generatorV(2).to(gpu)
optimizer=torch.optim.Adam([{'params':GV.parameters(),'lr':1e-3},{'params':z11,'lr':1e-2}])
#optimizer=torch.optim.Adam([{'params':GV.parameters(),'lr':1e-4}])
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=6, verbose=True, min_lr=1e-7)
pp=np.array([])
for ep1 in range(20000):#25000
    for bat in range(1):
        z1=torch.cat((z11,ur.unsqueeze(0).unsqueeze(0).cuda()),2)

        #z1=torch.reshape(V1,(1,1,nbasis,nf1))
        l1_reg=0.
        for param in GV.parameters():
            l1_reg += param.abs().sum()  
        v1=GV(z1)
        #v1=v1.permute(1,0)
        loss=abs(v1[0,0]-V1).pow(2).sum()#+200.0*KLloss(z11[0,0])+10.0*Smoothness(z11[0,0,:,:]) 
        if ep1%10==0:
            print(ep1,loss.item())
        
#        if ep1%5000==0:
#            plt.figure(ep1)
#            dd=(v1[0,0,:,20].unsqueeze(1).unsqueeze(1).unsqueeze(1)*recT.squeeze(1)).sum(dim=0).detach().cpu().numpy()
#            plt.imshow(np.abs((dd[0,150:350,150:300]+dd[1,150:350,150:300]*1j)),cmap='gray')
#            plt.show()
#            plt.pause(1)
            
#        if ep1%501==0:
#            plt.figure(ep1)
#            zz=z1.detach().cpu().numpy()
#            plt.plot(zz[0,0,0])
#            plt.plot(zz[0,0,1])
#            plt.show()
#            plt.pause(1)
    
#        dd=(v1[0,0,:,20:30].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).detach().cpu()*recT1[:,:,:,150:350,150:300].detach().cpu()).sum(dim=0).detach().cpu()
#        org1=(v2[0,0,:,20:30].unsqueeze(-1).unsqueeze(-1).detach().cpu()*recT[:,:,:,150:350,150:300].detach().cpu()).sum(dim=0)
#        psnr=torch.mean(myPSNR(org1,dd))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(ep1,loss.item())
        #pp=np.append(pp,psnr)
        #scheduler.step(loss.detach())
        #pp=np.append(pp,loss.item())        
#%%
z=recT#+torch.normal(0,0.03,(atb.shape[0],atb.shape[1],atb.shape[2],atb.shape[3])).cuda()#.permute(1,0,2,3)
z=torch.reshape(z,(1,2*nbasis,nx,nx))
#z1=torch.cuda.FloatTensor(1,1,nbasis,nf1).fill_(0)
#z1=torch.normal(0,1,(1,1,nbasis,nf1))
z1=torch.reshape(torch.tensor(V[:,0:NF]),(1,1,nbasis,NF))
#z1 = Variable(z1,requires_grad=True)






        #%%    
torch.cuda.empty_cache()
#from dn_modelV3 import SmallModel
#        G=SmallModel1().to(gpu)
#GV=SmallModel().to(gpu)
#        G.load_state_dict(torch.load('tempp_199.pt'))
##        G.load_state_dict(torch.load('wtsN-U3.pt'))

#z1=0.01*torch.ones((1,1,2,nf1)).to(gpu)
#z1[0,0,1,:]=torch.tensor(V[22,0:nf1])
#z1[0,0,0,:]=torch.tensor(V[27,0:nf1])
#z1 = Variable(z1,requires_grad=True)
##z = Variable(z,requires_grad=True)
#
#
G=SmallModel1().to(gpu)
#GV=SmallModel(16).to(gpu)
#
##GV.load_state_dict(torch.load('temppV2.pt')) 
##G.load_state_dict(torch.load('temppU4.pt')) 
#
#GV.load_state_dict(torch.load('temppV2_600.pt')) 
#G.load_state_dict(torch.load('temppU5_600.pt'))        
       
#        #optimizer=torch.optim.AdamW([{'params':G.parameters(),'lr':1e-4},{'params':GV.parameters(),'lr':1e-4},{'params':z,'lr':1e-4}])
#v1=(z1.cuda()).unsqueeze(-1)
#v1 = Variable(v1,requires_grad=True)
#optimizer=torch.optim.Adam([{'params':G.parameters(),'lr':1e-4},{'params':GV.parameters(),'lr':1e-4},{'params':z1,'lr':1e-4}])
#optimizer=torch.optim.Adam([{'params':G.parameters(),'lr':1e-4},{'params':GV.parameters(),'lr':1e-4}])

optimizer=torch.optim.Adam([{'params':G.parameters(),'lr':1e-4}])
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=6, verbose=True, min_lr=1e-5)

#optimizer=torch.optim.SGD([{'params':v1,'lr':1e-4}])#,{'params':z,'lr':5e-3,'momentum':0.9}])

smapT=smapT.cpu()
ktrajT=ktrajT.to(cpu)
dcf=torch.reshape(dcf,(nbasis,3,2,NF,nintl*nkx))
#plt.figure(2)
#dd=(np.reshape(V[:,26],(nbasis,1,1,1))*recT.detach().cpu().numpy()).sum(axis=0)
#plt.imshow(np.abs((dd[0,100:400,100:400]+dd[1,100:400,100:400]*1j)),cmap='gray')
pp=np.array([])
#plt.pause(1)
#plt.figure(1)
sf.tic()
for it in range(101):    
#            for bat in range(1):
        
    v1=(z1.cuda()).unsqueeze(-1)
    u1=G(z.cuda())#+z.cuda()
    u1=torch.reshape(u1,(nbasis,2,nx,nx))
    u1=u1.unsqueeze(1)
#          
    l1_regU=0.
    for param in G.parameters():
        l1_regU += param.abs().sum()
        
#    l1_regV=0.
#    for param in GV.parameters():
#        l1_regV += param.abs().sum()
    err=abs(AUV1(u1,v1)-kdataT[:,:,0:nf1].cuda()).pow(2)
    #loss=((dcf[0,:,:,0:nf1].pow(1))*(err)).sum()+0.0001*l1_regU+0.0001*l1_regV+1.0*Smoothness(z1[0,0,:,:])
    loss=((dcf[0,:,:,0:nf1].pow(1))*(err)).sum()#+0.000000*l1_regU+0.0000*l1_regV+1.0*Smoothness(z1[0,0,:,:])+100.0*KLloss(z1[0,0])

#            err=abs(yy1.squeeze(1)-atb).pow(2)          
#            loss=(err).sum()#+10*l1_reg#+0.1*(z.cpu()-u2.cpu()).pow(2).sum()+0.1*(z1[0,0].cpu()-v2.cpu()).pow(2).sum()
    if it%5==0:
        plt.figure(it)
        dd=(v1[0,0,:,20].unsqueeze(1).unsqueeze(1)*u1.squeeze(1)).sum(dim=0).detach().cpu().numpy()
        org2=(v1[0,0,:,20].unsqueeze(1).unsqueeze(1)*recT.squeeze(1)).sum(dim=0).detach().cpu().numpy()
        dd1=dd[0,150:350,150:300]+dd[1,150:350,150:300]*1j
        or1=org2[0,150:350,150:300]+org2[1,150:350,150:300]*1j
        dd1=np.concatenate((or1,dd1),axis=1)
        plt.imshow(np.abs((dd1)),cmap='gray')
        plt.show()
        plt.pause(1)           
#            if it%10==0:
#                torch.save(u1,'tmpU'+str(it)+'.pt')           
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step(loss.detach())
    
    dd=(v1[0,0,:,10:60].unsqueeze(-1).unsqueeze(-1).detach().cpu()*u1[:,:,:,150:350,150:300].detach().cpu()).sum(dim=0).detach().cpu()
    org2=(v1[0,0,:,10:60].unsqueeze(-1).unsqueeze(-1).detach().cpu()*recT[:,:,:,150:350,150:300].detach().cpu()).sum(dim=0)
    psnr=torch.mean(myPSNR(org2,dd))
    print(it,loss.item(),psnr.item())
    pp=np.append(pp,psnr)
sf.toc()
#    if ep1%2==0:
#        wtsFname1='wtsN-U'+str(ep1+1)+'.pt'
#        torch.save(G.state_dict(),wtsFname1) 
#        wtsFname2='wtsN-V'+str(ep1+1)+'.pt'
#        torch.save(GV.state_dict(),'temppV1.pt')      
#%%

#V=V.detach().cpu().numpy()
v1=v1.squeeze(0).squeeze(0)
Vx=np.squeeze(v1.detach().cpu().numpy())
Vz=V[:,50:nf1-10]#V.detach().cpu().numpy()

for i in range(20,30,1):
    plt.figure(i)
    plt.plot(Vz[i])
    plt.pause(0.1)
    plt.plot(Vx[i])
    #plt.pause()       
    plt.show() 
#    
# 
#u1=torch.load('tmpU29.pt')
u1=torch.reshape(u1,(nbasis,2,nx,nx))
rec = np.squeeze(u1.detach().cpu().numpy())
#rec=np.reshape(rec,(nbasis,nx,nx,2))
rec = rec[:,0] + 1j*rec[:,1]
        
plt.figure(1)
for i in range(30):
    plt.imshow((np.squeeze(np.abs(rec[i,100:400,100:400]))),cmap='gray')       
    plt.show()
    #plt.colorbar()
    plt.pause(1)
    
    
rec = np.squeeze(recT.detach().cpu().numpy())
#rec=np.reshape(rec,(nbasis,nx,nx,2))
rec = rec[:,0] + 1j*rec[:,1]
        
plt.figure(2)
for i in range(32):
    plt.imshow((np.squeeze(np.abs(rec[i,:,:,]))),cmap='gray')       
    plt.show()
    #plt.colorbar()
    plt.pause(1)
##
d2=sio.loadmat('/Shared/lss_jcb/abdul/IRdata/UbasisIR.mat')
rec=np.asarray(d2['x1'])  
#rec=np.reshape(rec,(nbasis,nx*nx))
rec1=rec@Vx
rec1=np.reshape(rec1,(nx,nx,nf1-60))
#sio.savemat('prpfinal_IR.mat',{"rec":rec1}) 
#rec1=np.fft.fftshift(np.fft.fftshift(rec1,0),1)
#plt.figure(1,[15,15])
for i in range(160):
    plt.imshow((np.squeeze(np.abs(rec1[:,:,i]))),cmap='gray')       
    plt.show() 
    plt.pause(0.01)
###    plt.pause(0.05)
###xx=np.fft.fftshift(np.fft.fftshift(atbV,1),2)
##rec=np.reshape(rec,(nbasis,nx*nx))
##rec1=rec.T@V[:,0:NF]
##rec1=np.reshape(rec1,(nx,nx,NF))
###rec1=np.fft.fftshift(np.fft.fftshift(rec1,0),1)
###plt.figure(1,[15,15])
##for i in range(100):
##    plt.imshow((np.squeeze(np.abs(rec1[:,:,i]))),cmap='gray')       
##    plt.show() 
##    plt.pause(0.05)
#
#rec=np.squeeze(atb.cpu().numpy())
#rec=rec[:,0]+rec[:,1]*1j
