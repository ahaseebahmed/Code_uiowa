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

from unetmodel import UnetClass
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


NF=300#100#900 
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
TF=300
thres=0.4
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
#G=UnetClass().to(gpu)
#G=SmallModel1().to(gpu)
#
#GV=SmallModel().to(gpu)
#G.load_state_dict(torch.load('wtsN-U3.pt'))
#GV.load_state_dict(torch.load('wtsN-V3.pt'))
##GV.load_state_dict(torch.load('./PTmodels/27Oct_112451am_500ep_27oct/wts-500.pt'))
#
##optimizer=torch.optim.SGD([{'params':G.parameters(),'lr':5e-3,'momentum':0.9}])
##optimizer=torch.optim.AdamW([{'params':G.parameters(),'lr':1e-4}])
#
#optimizer=torch.optim.AdamW([{'params':G.parameters(),'lr':1e-4},{'params':GV.parameters(),'lr':1e-4},{'params':z,'lr':1e-4}])
#scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=6, verbose=True, min_lr=1e-6)

trnFiles=os.listdir('/Shared/lss_jcb/abdul/prashant_cardiac_data/Data/d2/')
sz=len(trnFiles)
#data2=np.zeros((sz,1,n_select,N,N)).astype(np.complex64)

rndm=random.sample(range(sz),sz)
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
ktraj=ktraj[0:NF]
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
#cc[0,0,:,236,247]=1

tr=ktrajT[0].unsqueeze(0)
dd=nuf_ob(cc,tr.cuda())
ee1=adjnuf_ob(dd*dcf.cuda(),tr.cuda())
dcf=dcf/ee1[0,0,0,255,255]


dcf=torch.reshape(dcf,(2,NF,nintl,nx))
dcf[:,:,0]=0
dcf[:,:,1]=0
dcf=torch.reshape(dcf,(1,1,2,NF*nintl*nx))
#real_mat, imag_mat = precomp_sparse_mats(ktrajT, adjnuf_ob)
#interp_mats = {
#    'real_interp_mats': real_mat,
#    'imag_interp_mats': imag_mat
#}

#dcf=dcf+0.01
ktrajT_ch=ktrajT[0].unsqueeze(0)
dcf=dcf.repeat(nbasis,nch,1,1)
#df=torch.reshape(dcf[0,0,0],(NF,nintl*nx))
#df=df.unsqueeze(1).unsqueeze(1)
#df=df.repeat(1,3,2,1)
#df2=torch.reshape(df,(NF,3,2,nintl,nx))
#df2[:,:,:,0]=0
#df2[:,:,:,1]=0
#
#df2=torch.reshape(df2,(NF,3*2*nintl*nx))
#%%
for ep1 in range(10):
    rndm=random.sample(range(sz),sz)
    for fl in range(1):
        #%%
        fl=0
        str1=trnFiles[fl].split('_kd.mat')[0]
        #dictn=sio.loadmat('/Shared/lss_jcb/abdul/prashant_cardiac_data/Data/'+trnFiles[rndm[fl]])
        #d1=sio.loadmat('/Shared/lss_jcb/abdul/prashant_cardiac_data/Data/d2/'+str1+'_kd.mat')
        #dictn=sio.loadmat('/Shared/lss_jcb/abdul/prashant_cardiac_data/Data/d1/'+str1+'.mat')
        dictn=sio.loadmat('/Shared/lss_jcb/abdul/prashant_cardiac_data/Data/meas_MID00185_FID142454_SAX_gNAV_gre11.mat')
        d1=sio.loadmat('/Shared/lss_jcb/abdul/prashant_cardiac_data/Data/kdata.mat')
        kdata=np.asarray(d1['kdata'])
        
        kdata[:,:,0]=0
        kdata[:,:,1]=0
        kdata=np.transpose(kdata,(3,1,2,0))
        kdata=kdata[0:NF,0:nch]
        kdata=np.reshape(kdata,(NF,nch*nintl*nx))
        kdata=kdata/2.9754e-5
        #kdata=kdata/np.max(np.abs(kdata))
        
        L=np.asarray(dictn['L']) 
        L=L.astype(np.complex64)
        L=L[0:NF,0:NF]
        U,sb,V=np.linalg.svd(L)
        #sb[NF-1]=0
        V=V[TF-nbasis:TF,0+st:NF+st]
        
        sb=np.diag(sb[NF-nbasis:NF,])*lam
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
        kdataT=torch.reshape(kdataT,(NF,nch,2,nx*nintl))
        kdataT=kdataT.permute(1,2,0,3)
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
        csmTrn = espirit(x_f, 6, 24, 0.05, 0.9925)
        csm=csmTrn[:,:,0,:,0]
        csm=np.transpose(csm,(2,0,1))
    
        smap = np.stack((np.real(csm), np.imag(csm)), axis=1)
        smap=np.tile(smap,(nbasis,1,1,1,1))
        

        
        cimg=np.sum(np.abs(np.squeeze(x)),axis=2)
        cimg1=ndimage.convolve(cimg,np.ones((13,13)))
        W=cimg1/np.max(np.abs(cimg1))
        thres=0.3
        W[W<thres]=0;
        W[W>=thres]=1;
        W=W.astype(int)
  
#W1=ndimage.morphology.binary_closing(W,structure=np.ones((3,3)))
        W1=ndimage.morphology.binary_fill_holes(W)
        W1=W1.astype(float)
        W1=ndimage.convolve(W1,np.ones((13,13)))
        W1=W1/np.max(W1)
        #W1=(1-W1)
        #W=np.tile(W1,(nbasis,2,1,1))
        W1=np.reshape(W1,(1,1,1,nx,nx))
        #smap=W1*smap
        smapT = torch.tensor(smap).to(dtype)
        smapT=smapT.to(gpu)
        
        del coilimages               
#%%     
        #wt=np.array(range(1,31))/30
        #wt=torch.tensor(wt).to(dtype)
        
        nufft_ob = MriSenseNufft(im_size=im_size,smap=smapT).to(dtype)
        nufft_ob=nufft_ob.to(gpu)
        
        adjnufft_ob = AdjMriSenseNufft(im_size=im_size, smap=smapT).to(dtype)
        adjnufft_ob=adjnufft_ob.to(gpu)
        
        atb=torch.zeros((nbasis,im_size[0],im_size[1]))
        
        # generating AHb results
        #dcf=torch.tensor(dcf).unsqueeze(0).unsqueeze(0)
        #dcf=dcf.repeat(1,1,2,1)
        #dcf=dcf*2
        atb=adjnufft_ob(kdaT*2*dcf.cuda(),ktrajT.cuda())
        atb=atb.squeeze(1)
        #atb=atb*torch.tensor(W).cuda()
        mx=torch.max((atb**2).sum(dim=-3).sqrt())
        mx1=0.0073#torch.max((yy1**2).sum(dim=-3).sqrt())
        
#        kdata_ch=torch.reshape(kdata_ch,(1,3,2,NF,nintl*nx))
#        ktrajt=torch.reshape(ktrajT,(nbasis,2,NF,nintl*nx))
#        dcf=torch.reshape(dcf,(nbasis,3,2,NF,nintl*nx))
#        #z1=torch.cuda.FloatTensor(3,2*nbasis,nx,nx).fill_(0)
#        #z2=torch.cuda.FloatTensor(3,1,nbasis,nf1).fill_(0) 
#        itt=0
#        nf1=300
#        v1=np.reshape(V[:,0:nf1],(nbasis,1,1,nf1,1))
#        #v1=torch.reshape(v1.squeeze(0),(nbasis,1,1,nf1,1))
#        
#        kdaT1=kdata_ch[:,:,:,itt*nf1:nf1*(itt+1)]
#        ktrajt1=ktrajt[:,:,itt*nf1:nf1*(itt+1)]
#        dcf1=dcf[:,:,:,itt*nf1:nf1*(itt+1)]
##            z2=VT[:,itt*nf1:nf1*(itt+1)].unsqueeze(0).unsqueeze(0).cuda()
#        kdaT1=kdaT1*torch.tensor(v1).cuda()
#        kdaT1=torch.reshape(kdaT1,(nbasis,nch,2,nf1*nintl*nx))
#        dcf1=torch.reshape(dcf1,(nbasis,nch,2,nf1*nintl*nx))
#        ktrajt1=torch.reshape(ktrajt1,(nbasis,2,nf1*nintl*nx))
#
#        atb1=adjnufft_ob(kdaT1*2*dcf1.cuda(),ktrajt1.cuda())
#        atb1=atb1.squeeze(1)
        #atb=atb*torch.tensor(W).cuda()
            #mx=torch.max((atb1**2).sum(dim=-3).sqrt())
        #torch.max((yy1**2).sum(dim=-3).sqrt())
        #mx=torch.max(atb)
            #atb1=atb1/mx
            #z=atb
        #atb1=torch.reshape(atb1,(1,2*nbasis,nx,nx))
        #mx=torch.max(atb)
        #atb=atb/mx
#        bb=torch.reshape(atb,(nbasis,2*nx*nx))
#        V0=torch.tensor(V)
#        #V1=V.T@np.linalg.pinv(V@V.T)
#        xx=V0.T.cuda()@bb
#        pp=torch.tensor(V1).cuda()@xx
#        pp=torch.reshape(pp,(nbasis,2,nx,nx))
#        #atb=atb.to(dtype)
#        kdataT=kdataT/mx
        
#        kdaT1=kdaT/mx
#        atb=adjnufft_ob(kdaT1*dcf.cuda(),ktrajT.cuda())
#        atb=atb.squeeze(1)
#        #atb=atb*torch.tensor(W).cuda()
#        mx1=torch.max((atb**2).sum(dim=-3).sqrt())
        #mx=torch.max(atb)
        #dcomp = calculate_radial_dcomp_pytorch(nufft_ob, adjnufft_ob, xx.cuda()).unsqueeze(0).unsqueeze(0)
#        toep_ob = ToepSenseNufft(smap=smapT)
#        dcomp_kern = calc_toep_kernel(adjnufft_ob, ktrajT.cuda(), weights=dcf.cuda())
#        image_sharp_toep = toep_ob(atb, dcomp_kern)
        #G=UnetClass().to(gpu)
        G=SmallModel1().to(gpu)
        optimizer=torch.optim.Adam([{'params':G.parameters(),'lr':1e-4}])
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=6, verbose=True, min_lr=1e-5)
        z=atb
        z=torch.reshape(z,(1,2*nbasis,nx,nx)).to(gpu)
        z1=torch.cuda.FloatTensor(3,2*nbasis,nx,nx).fill_(0) 
        #noi=[0.0003,0.05,0.01]
        #noi=[0.003,0.0005,0.01]
        #noi=[0.05,0.0005,0.01]
        noi=[0.05,0.01,0.1]

        for it in range(3):
            z1[it]=z+torch.normal(0,noi[it],(z.shape[0],z.shape[1],z.shape[2],z.shape[3])).cuda()
        z=z.repeat(3,1,1,1)
        sf.tic()
        for rn in range(2000):
            #z=atb#+torch.normal(0,noi[it],(atb.shape[0],atb.shape[1],atb.shape[2],atb.shape[3])).cuda()
            #z=torch.reshape(z,(1,2*nbasis,nx,nx))
                #z=atb+torch.normal(0,0.005,(atb.shape[0],atb.shape[1],atb.shape[2],atb.shape[3])).cuda()
                #z=torch.reshape(z,(1,2*nbasis,nx,nx))
            u1=G(z)#+z.cuda()
            #u1=torch.reshape(u1,(nbasis,2,nx,nx))
            loss=((abs(u1-z))).pow(2).sum()
            if rn%10==0:
                print(rn,loss.item())
            #torch.cuda.empty_cache()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #scheduler.step(loss.detach())
        sf.toc()
#        
#        sf.tic()
#        for ep1 in range(500):
#            for bat in range(1):
#                V1=torch.tensor(V).cuda()+torch.normal(0,0.01,(V.shape[0],V.shape[1])).cuda()
#                z1=torch.reshape(V1,(1,1,nbasis,NF))
#                v1=GV(z1)
#                #v1=v1.permute(1,0)
#                loss=abs(v1[0,0]-torch.tensor(V).cuda()).pow(2).sum()
#                if ep1%10==0:
#                    print(ep1,loss.item())
#                
#                optimizer.zero_grad()
#                loss.backward()
#                optimizer.step()
#                scheduler.step(loss.detach())
#        sf.toc()  

        
        del kdaT,smap,adjnufft_ob
        #%%
        z=atb#+torch.normal(0,0.03,(atb.shape[0],atb.shape[1],atb.shape[2],atb.shape[3])).cuda()#.permute(1,0,2,3)
        z=torch.reshape(z,(1,2*nbasis,nx,nx))
        #z=torch.cuda.FloatTensor(1,2*nbasis,nx,nx).fill_(0) 
        z1=torch.reshape(torch.tensor(V),(1,1,nbasis,NF))
        #z = Variable(z,requires_grad=True)
        #%%    
        torch.cuda.empty_cache()
        #from dn_modelU2 import SmallModel1
        G=SmallModel1().to(gpu)
        GV=SmallModel().to(gpu)
        G.load_state_dict(torch.load('tempp_199.pt'))
#        G.load_state_dict(torch.load('wtsN-U3.pt'))
#        GV.load_state_dict(torch.load('wtsN-V3.pt'))        
        #optimizer=torch.optim.AdamW([{'params':G.parameters(),'lr':1e-4},{'params':GV.parameters(),'lr':1e-4},{'params':z,'lr':1e-4}])
        optimizer=torch.optim.Adam([{'params':G.parameters(),'lr':1e-4}])#,{'params':z,'lr':1e-4}])
        #optimizer=torch.optim.AdamW([{'params':G.parameters(),'lr':1e-4},{'params':GV.parameters(),'lr':1e-4}])
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=6, verbose=True, min_lr=1e-5)
        
        #optimizer=torch.optim.SGD([{'params':G.parameters(),'lr':1e-4}])#,{'params':z,'lr':5e-3,'momentum':0.9}])
        
        smapT=smapT.cpu()
        ktrajT=ktrajT.to(cpu)
        plt.figure(2)
        dd=(np.reshape(V[:,26],(nbasis,1,1,1))*atb.detach().cpu().numpy()).sum(axis=0)
        plt.imshow(np.abs((dd[0,100:400,100:400]+dd[1,100:400,100:400]*1j)),cmap='gray')
        plt.pause(1)
        plt.figure(1)
        sf.tic()
        for it in range(500):    
#            for bat in range(1):
                
            v1=(z1.cuda()).unsqueeze(-1)
            u1=G(z.cuda())#+z.cuda()
            u1=torch.reshape(u1,(nbasis,2,nx,nx))
            u1=u1.unsqueeze(1)
            yy=torch.cuda.FloatTensor(nch,2,NF,nx*nintl).fill_(0)
            for i in range(nbasis):
                xx=nufft_ob(u1[i].unsqueeze(0),ktrajT[i].unsqueeze(0).cuda(),smap=smapT[i].unsqueeze(0).cuda())
                xx=torch.reshape(xx,(nch,2,NF,nx*nintl))
                yy=yy+v1[:,:,i]*xx                    
            
            #err=abs(yy-kdataT.cuda()).pow(2)
            #loss=(dcf[0]*(err)).sum()
            
            yy1=torch.cuda.FloatTensor(nbasis,1,2,nx,nx).fill_(0) 
            for i in range(nbasis):                    
                tmp=v1[:,:,i]*yy
                tmp=torch.reshape(tmp,(1,nch,2,NF*nintl*nx))
                yy1[i]=adjnufft_ob(tmp*500*(dcf[i]).unsqueeze(0).cuda(),ktrajT[i].unsqueeze(0).cuda(),smap=smapT[i].unsqueeze(0).cuda())
            
            
            err=abs(yy1.squeeze(1)-atb).pow(2)
            l1_reg=0.
            for param in G.parameters():
                l1_reg += param.abs().sum()
            
            loss=(err).sum()#+10*l1_reg#+0.1*(z.cpu()-u2.cpu()).pow(2).sum()+0.1*(z1[0,0].cpu()-v2.cpu()).pow(2).sum()
            if it%1==0:
                print(it,loss.item())
                dd=(v1[0,0,:,26].unsqueeze(1).unsqueeze(1)*u1.squeeze(1)).sum(dim=0).detach().cpu().numpy()
                plt.imshow(np.abs((dd[0,100:400,100:400]+dd[1,100:400,100:400]*1j)),cmap='gray')
                plt.show()
                plt.pause(1)
            

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step(loss.detach())
        sf.toc()
#    if ep1%2==0:
#        wtsFname1='wtsN-U'+str(ep1+1)+'.pt'
#        torch.save(G.state_dict(),wtsFname1) 
#        wtsFname2='wtsN-V'+str(ep1+1)+'.pt'
#        torch.save(GV.state_dict(),wtsFname2)      
#%%

#V=V.detach().cpu().numpy()
v1=v1.squeeze(0).squeeze(0)
Vx=np.squeeze(v1.detach().cpu().numpy())
Vz=V#V.detach().cpu().numpy()

for i in range(29,20,-1):
    plt.figure(i)
    plt.plot(Vz[i])
    plt.pause(0.1)
    plt.plot(Vx[i])
    #plt.pause()       
    plt.show() 
#    
# 
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
    
    
rec = np.squeeze(atb.detach().cpu().numpy())
#rec=np.reshape(rec,(nbasis,nx,nx,2))
rec = rec[:,0] + 1j*rec[:,1]
        
plt.figure(2)
for i in range(30):
    plt.imshow((np.squeeze(np.abs(rec[i,100:400,100:400]))),cmap='gray')       
    plt.show()
    #plt.colorbar()
    plt.pause(1)
##    
rec=np.reshape(rec,(nbasis,nx*nx))
rec1=rec.T@V[:,0:NF]
rec1=np.reshape(rec1,(nx,nx,NF))
#sio.savemat('tmp11_dnnufft1.mat',{"rec":rec1}) 
#rec1=np.fft.fftshift(np.fft.fftshift(rec1,0),1)
#plt.figure(1,[15,15])
for i in range(50):
    plt.imshow((np.squeeze(np.abs(rec1[100:400,100:400,i]))),cmap='gray')       
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
