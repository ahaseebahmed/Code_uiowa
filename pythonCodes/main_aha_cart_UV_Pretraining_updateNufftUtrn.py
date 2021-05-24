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
nintl=10
#N=400
N1=300
N2=50
nch=3
thres=0.05
nbasis=30
lam=0.01
st=0
batch_sz=100
TF=900
im_size = [nx,nx]

#%%

#def for_nufft(nuff,x,csm,traj):
#def for_nufft(nufft_ob,x,csm,traj):
#    nch=csm.shape[1]
#    nbas=x.shape[0]
#    tmp1=torch.zeros((nch,nbas,2,NF*nx*nintl)).cuda()
#    for i in range(nch):
#        tmp=sf.complex_mult(x,csm[:,i].repeat(nbas,1,1,1,1),dim=2)
#        tmp=torch.reshape(tmp,(nbas,1,2,nx,nx))
#        #xx=ktrajT[0].unsqueeze(0)
#        tmp1[i]=nufft_ob(tmp,traj.repeat(nbas,1,1).cuda()).squeeze(1)
#    tmp1=tmp1.permute(1,0,2,3)
#    return tmp1
    
#%%
##G=UnetClass().to(gpu)
#G=SmallModel1().to(gpu)
#
#GV=SmallModel().to(gpu)
##G.load_state_dict(torch.load('wts-10U2.pt'))
##GV.load_state_dict(torch.load('wts-10V2.pt'))
##GV.load_state_dict(torch.load('./PTmodels/27Oct_112451am_500ep_27oct/wts-500.pt'))
#
##optimizer=torch.optim.SGD([{'params':G.parameters(),'lr':5e-3,'momentum':0.9}])
##optimizer=torch.optim.AdamW([{'params':G.parameters(),'lr':1e-4}])
#
#optimizer=torch.optim.AdamW([{'params':G.parameters(),'lr':1e-4},{'params':GV.parameters(),'lr':1e-4}])
#scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=6, verbose=True, min_lr=1e-5)
#


trnFiles=os.listdir('/Shared/lss_jcb/abdul/prashant_cardiac_data/Data/d2/')
sz=len(trnFiles)
#data2=np.zeros((sz,1,n_select,N,N)).astype(np.complex64)

rndm=random.sample(range(sz),sz)

Ub=torch.zeros(sz,2*nbasis,nx,nx)
Vb=torch.zeros(sz,1,nbasis,NF)
#%%
#nuf_ob = KbNufft(im_size=(nx,nx)).to(dtype)
#nuf_ob=nuf_ob.to(gpu)
#
#adjnuf_ob = AdjKbNufft(im_size=(nx,nx)).to(dtype)
#adjnuf_ob=adjnuf_ob.to(gpu)

smapT=torch.ones((1,1,2,nx,nx)).cuda()

nuf_ob = MriSenseNufft(im_size=(nx,nx),smap=smapT).to(dtype)
nuf_ob=nuf_ob.to(gpu)

adjnuf_ob = AdjMriSenseNufft(im_size=(nx,nx), smap=smapT).to(dtype)
adjnuf_ob=adjnuf_ob.to(gpu)



d2=sio.loadmat('/Shared/lss_jcb/abdul/prashant_cardiac_data/Data/ktraj.mat')
ktraj=np.asarray(d2['ktraj'])
ktraj=ktraj.astype(np.complex64)
ktraj=np.transpose(ktraj,(2,1,0))/nx
ktraj=np.reshape(ktraj,(1,NF*nintl*nx))*2*np.pi
#dcf=np.abs(ktraj)

ktraj = np.stack((np.real(ktraj), np.imag(ktraj)), axis=1)
dcf = calculate_radial_dcomp_pytorch(nuf_ob, adjnuf_ob, torch.tensor(ktraj).cuda())
dcf=torch.tensor(dcf).unsqueeze(0).unsqueeze(0)
#dcf=dcf.repeat(nbasis,nch,2,1)
dcf=dcf.repeat(1,1,2,1)


ktraj=np.tile(ktraj,(nbasis,1,1))
ktrajT = torch.tensor(ktraj).to(dtype)


cc=torch.zeros((1,1,2,nx,nx)).cuda()
cc[0,0,:,255,255]=1
tr=ktrajT[0].unsqueeze(0)
dd=nuf_ob(cc,tr.cuda())
ee1=adjnuf_ob(dd*dcf.cuda(),tr.cuda())
dcf=dcf/ee1[0,0,0,255,255]

#real_mat, imag_mat = precomp_sparse_mats(ktrajT, adjnuf_ob)
#interp_mats = {
#    'real_interp_mats': real_mat,
#    'imag_interp_mats': imag_mat
#}

#dcf=dcf+0.01
ktrajT_ch=ktrajT[0].unsqueeze(0)
dcf=dcf.repeat(nbasis,nch,1,1)


#%%
for ep1 in range(1):
    rndm=random.sample(range(sz),sz)
    for fl in range(sz):
        #%%
        print(fl)
        str1=trnFiles[fl].split('_kd.mat')[0]
        #dictn=sio.loadmat('/Shared/lss_jcb/abdul/prashant_cardiac_data/Data/'+trnFiles[rndm[fl]])
        d1=sio.loadmat('/Shared/lss_jcb/abdul/prashant_cardiac_data/Data/d2/'+str1+'_kd.mat')
        dictn=sio.loadmat('/Shared/lss_jcb/abdul/prashant_cardiac_data/Data/d1/'+str1+'.mat')
        #dictn=sio.loadmat('/Shared/lss_jcb/abdul/prashant_cardiac_data/Data/meas_MID00185_FID142454_SAX_gNAV_gre11.mat')
        #d1=sio.loadmat('/Shared/lss_jcb/abdul/prashant_cardiac_data/Data/kdata.mat')
        kdata=np.asarray(d1['kdata'])
        kdata[:,:,0]=0
        kdata[:,:,1]=0
        kdata=np.transpose(kdata,(3,1,2,0))
        kdata=kdata[:,0:nch]
        kdata=np.reshape(kdata,(NF,nch*nintl*nx))
        #kdata=kdata/np.max(np.abs(kdata))
        
        L=np.asarray(dictn['L']) 
        L=L.astype(np.complex64)
        U,sb,V=np.linalg.svd(L)
        sb[NF-1]=0
        V=V[TF-nbasis:TF,0+st:NF+st]
        sb=np.diag(sb[NF-nbasis:NF,])*lam
        V=V.astype(np.float32)
        Vn=np.zeros((nbasis,NF,NF),dtype='float32')  
        for i in range(nbasis):
            Vn[i,:,:]=np.diag(V[i,:])
        Vn=np.reshape(Vn,(nbasis*NF,NF))
        temp=Vn@kdata
        temp=np.reshape(temp,(nbasis,NF,nch,nintl*nx))        
        temp=np.transpose(temp,(0,2,1,3))
        temp=np.reshape(temp,(nbasis,nch,NF*nintl*nx))
        kdata=np.reshape(kdata,(NF,nch,nintl*nx))
        
        
        #Tensors
        #csmTrn[0].shape
        kdata = np.stack((np.real(kdata), np.imag(kdata)),axis=2)
        kda1 = np.stack((np.real(temp), np.imag(temp)),axis=2)

        # convert to tensor, unsqueeze batch and coil dimension
        kdataT = torch.tensor(kdata).to(dtype)
        kdata_ch=kdataT.permute(1,2,0,3)
        kdataT=torch.reshape(kdataT,(NF,nch*2*nx*nintl))
        kdaT = torch.tensor(kda1).to(dtype)
        kdaT=kdaT.to(gpu)       
        
#%      Coilimages ans csm estimation  
        
        kdata_ch=torch.reshape(kdata_ch,(nch,2,NF*nintl*nx))
        kdata_ch=kdata_ch.unsqueeze(0)
        kdata_ch=kdata_ch.to(gpu)
        ktrajT_ch=ktrajT_ch.to(gpu)
        #dcfT=dcfT.to(gpu)        
        #nuf_ob = KbNufft(im_size=im_size).to(dtype)
        #nuf_ob=nuf_ob.to(gpu)
        
        adjnuf_ob=adjnuf_ob.to(gpu)
        
        coilimages=torch.zeros((1,nch,2,nx,nx))
#        A=lambda x: nuf_ob(x,ktrajT)
#        At=lambda x:adjnuf_ob(x,ktrajT)
#        AtA=lambda x:At(A(x))
        for i in range(nch):
            coilimages[:,i]=adjnuf_ob(kdata_ch[:,i].unsqueeze(1),ktrajT_ch)
            #ini=torch.zeros_like(temp)
            #coilimages[:,i]=sf.pt_cg(AtA,temp,ini,50,1e-15)
        X=coilimages.cpu().numpy()
        x=X[:,:,0]+X[:,:,1]*1j
        x=np.transpose(x,(2,3,0,1))
        x_f = fft(x, (0, 1, 2))
        csmTrn = espirit(x_f, 6, 24, 0.03, 0.9925)
        csm=csmTrn[:,:,0,:,0]
        csm=np.transpose(csm,(2,0,1))
    
        smap = np.stack((np.real(csm), np.imag(csm)), axis=1)
        smap=np.tile(smap,(nbasis,1,1,1,1))
        smapT = torch.tensor(smap).to(dtype)
        smapT=smapT.to(gpu)
        
        del kdata_ch,coilimages               
        
        
        adjnufft_ob = AdjMriSenseNufft(im_size=im_size, smap=smapT).to(dtype)
        adjnufft_ob=adjnufft_ob.to(gpu)
        
        atb=torch.zeros((nbasis,im_size[0],im_size[1]))
        
        # generating AHb results
        #dcf=torch.tensor(dcf).unsqueeze(0).unsqueeze(0)
        #dcf=dcf.repeat(1,1,2,1)
        atb=adjnufft_ob(kdaT*dcf.cuda(),ktrajT.cuda())
        atb=atb.squeeze(1)
        mx=torch.max((atb**2).sum(dim=-3).sqrt())
        #mx=torch.max(atb)
        atb=atb/mx
        z=torch.reshape(atb,(1,2*nbasis,nx,nx))
        Ub[fl]=z.cpu()
        Vb[fl]=torch.reshape(torch.tensor(V),(1,1,nbasis,NF)).cpu()
        del atb,smapT
        torch.cuda.empty_cache()
        #kdataT=kdataT/mx
        
        #dcomp = calculate_radial_dcomp_pytorch(nufft_ob, adjnufft_ob, xx.cuda()).unsqueeze(0).unsqueeze(0)
#        toep_ob = ToepSenseNufft(smap=smapT)
#        dcomp_kern = calc_toep_kernel(adjnufft_ob, ktrajT.cuda(), weights=dcf.cuda())
#        image_sharp_toep = toep_ob(atb, dcomp_kern)
        #z=atb+torch.normal(0,0.01,(atb.shape[0],atb.shape[1],atb.shape[2],atb.shape[3])).cuda()
#        #z=torch.reshape(z,(1,2*nbasis,nx,nx))
#        for it in range(500):
#            z=atb+torch.normal(0,0.005,(atb.shape[0],atb.shape[1],atb.shape[2],atb.shape[3])).cuda()
#            z=torch.reshape(z,(1,2*nbasis,nx,nx))
#            u1=G(z.cuda())
#            u1=torch.reshape(u1,(nbasis,2,nx,nx))
#            loss=abs(u1-atb).pow(2).sum()
#            print(it,loss.item())
#            #torch.cuda.empty_cache()
#            optimizer.zero_grad()
#            loss.backward()
#            optimizer.step()
#            scheduler.step(loss.detach())
        
        
        #%%


#    if ep1%2==0:
#        wtsFname1='wts-10U'+str(ep1+1)+'.pt'
#        torch.save(G.state_dict(),wtsFname1) 
#        wtsFname2='wts-10V'+str(ep1+1)+'.pt'

