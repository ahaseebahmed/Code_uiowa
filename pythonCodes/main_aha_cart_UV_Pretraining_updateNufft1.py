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
G=UnetClass().to(gpu)
G=SmallModel1().to(gpu)

GV=SmallModel().to(gpu)
#G.load_state_dict(torch.load('wts-10U2.pt'))
#GV.load_state_dict(torch.load('wts-10V2.pt'))
#GV.load_state_dict(torch.load('./PTmodels/27Oct_112451am_500ep_27oct/wts-500.pt'))

#optimizer=torch.optim.SGD([{'params':G.parameters(),'lr':5e-3,'momentum':0.9}])
optimizer=torch.optim.AdamW([{'params':G.parameters(),'lr':1e-3}])

#optimizer=torch.optim.AdamW([{'params':G.parameters(),'lr':1e-3},{'params':GV.parameters(),'lr':1e-3}])
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=6, verbose=True, min_lr=5e-5)

trnFiles=os.listdir('/Shared/lss_jcb/abdul/prashant_cardiac_data/Data/d2/')
sz=len(trnFiles)
#data2=np.zeros((sz,1,n_select,N,N)).astype(np.complex64)

rndm=random.sample(range(sz),sz)
#%%
nuf_ob = KbNufft(im_size=(nx,nx),norm='ortho').to(dtype)
nuf_ob=nuf_ob.to(gpu)

adjnuf_ob = AdjKbNufft(im_size=(nx,nx),norm='ortho').to(dtype)
adjnuf_ob=adjnuf_ob.to(gpu)

smapT=torch.ones((1,1,2,nx,nx)).cuda()

nuf_ob = MriSenseNufft(im_size=(nx,nx),smap=smapT,norm='ortho').to(dtype)
nuf_ob=nuf_ob.to(gpu)

adjnuf_ob = AdjMriSenseNufft(im_size=(nx,nx), smap=smapT, norm='ortho').to(dtype)
adjnuf_ob=adjnuf_ob.to(gpu)



d2=sio.loadmat('/Shared/lss_jcb/abdul/prashant_cardiac_data/Data/ktraj.mat')
ktraj=np.asarray(d2['ktraj'])
ktraj=ktraj.astype(np.complex64)
ktraj=np.transpose(ktraj,(2,1,0))/nx
ktraj=np.reshape(ktraj,(1,NF*nintl*nx))*2*np.pi
dcf=np.abs(ktraj)
dcf=torch.tensor(dcf).unsqueeze(0).unsqueeze(0)
#dcf=dcf.repeat(nbasis,nch,2,1)
dcf=dcf.repeat(1,1,2,1)
ktraj = np.stack((np.real(ktraj), np.imag(ktraj)), axis=1)
#dcf = calculate_radial_dcomp_pytorch(nuf_ob, adjnuf_ob, torch.tensor(ktraj).cuda())

ktraj=np.tile(ktraj,(nbasis,1,1))
ktrajT = torch.tensor(ktraj).to(dtype)


cc=torch.zeros((1,1,2,nx,nx)).cuda()
cc[0,0,:,255,255]=1
tr=ktrajT[0].unsqueeze(0)
dd=nuf_ob(cc,tr.cuda(),smap=smapT[0].unsqueeze(0))
ee1=adjnuf_ob(dd*dcf.cuda(),tr.cuda(),smap=smapT[0].unsqueeze(0))
dcf=dcf/ee1[0,0,0,255,255]

#real_mat, imag_mat = precomp_sparse_mats(ktrajT, adjnuf_ob)
#interp_mats = {
#    'real_interp_mats': real_mat,
#    'imag_interp_mats': imag_mat
#}

#dcf=dcf+0.01
ktrajT_ch=ktrajT[0].unsqueeze(0)

#%%
for ep1 in range(10):
    rndm=random.sample(range(sz),sz)
    for fl in range(sz):
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
        
        
        #%%
        #csmTrn[0].shape
        kdata = np.stack((np.real(kdata), np.imag(kdata)),axis=2)
        kda1 = np.stack((np.real(temp), np.imag(temp)),axis=2)

        # convert to tensor, unsqueeze batch and coil dimension
        kdataT = torch.tensor(kdata).to(dtype)
        kdata_ch=kdataT.permute(1,2,0,3)
        kdataT=torch.reshape(kdataT,(NF,nch*2*nx*nintl))
        kdaT = torch.tensor(kda1).to(dtype)
        kdaT=kdaT.to(gpu)       
#%%        
        
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
        csmTrn = espirit(x_f, 6, 24, 0.01, 0.9925)
        csm=csmTrn[:,:,0,:,0]
        csm=np.transpose(csm,(2,0,1))
    
        smap = np.stack((np.real(csm), np.imag(csm)), axis=1)
        smap=np.tile(smap,(nbasis,1,1,1,1))
        smapT = torch.tensor(smap).to(dtype)
        smapT=smapT.to(gpu)
        
        del kdata_ch,coilimages               
#%%        
        
        adjnufft_ob = AdjMriSenseNufft(im_size=im_size, smap=smapT, norm='ortho').to(dtype)
        adjnufft_ob=adjnufft_ob.to(gpu)
        
        atb=torch.zeros((nbasis,im_size[0],im_size[1]))
        
        # generating AHb results
        #dcf=torch.tensor(dcf).unsqueeze(0).unsqueeze(0)
        dcf=dcf.repeat(nbasis,nch,1,1)
        #dcf=dcf.repeat(1,1,2,1)
        atb=adjnufft_ob(kdaT*dcf.cuda(),ktrajT.cuda())
        atb=atb.squeeze(1)
        
        #dcomp = calculate_radial_dcomp_pytorch(nufft_ob, adjnufft_ob, xx.cuda()).unsqueeze(0).unsqueeze(0)
#        toep_ob = ToepSenseNufft(smap=smapT)
#        dcomp_kern = calc_toep_kernel(adjnufft_ob, ktrajT.cuda(), weights=dcf.cuda())
#        image_sharp_toep = toep_ob(atb, dcomp_kern)


        
        
        nufft_ob = MriSenseNufft(im_size=im_size,smap=smapT,norm='ortho').to(dtype)
        nufft_ob=nufft_ob.to(gpu)
        
#        adjnufft_ob = AdjMriSenseNufft(im_size=im_size, smap=smapT, norm='ortho').to(dtype)
#        adjnufft_ob=adjnufft_ob.to(gpu)
#        
#        smapT=torch.ones((1,1,2,nx,nx)).cuda()
#        aa=atb[29].unsqueeze(0).unsqueeze(0)
#        tr=ktrajT[0].unsqueeze(0)
#        xx=nufft_ob(aa,tr.cuda(),smap=smapT[0].unsqueeze(0))
#        bb=adjnufft_ob(xx*dcf.cuda(),tr.cuda(),smap=smapT[0].unsqueeze(0))
#        
#        dcf1=dcf/(torch.sum(aa*bb)/torch.sum(aa*aa))      
#        bb1=adjnufft_ob(xx*dcf1.cuda(),tr.cuda(),smap=smapT[0].unsqueeze(0))
#        
#        cc=torch.ones((1,1,2,nx,nx)).cuda()
#        tr=ktrajT[0].unsqueeze(0)
#        dd=nufft_ob(cc,tr.cuda(),smap=smapT[0].unsqueeze(0))
#        ee=adjnufft_ob(dd*dcf.cuda(),tr.cuda(),smap=smapT[0].unsqueeze(0))
#
#
#        dcf1=dcf/3.4109
#        cc=torch.zeros((1,1,2,nx,nx)).cuda()
#        cc[0,0,:,255,255]=1
#        tr=ktrajT[0].unsqueeze(0)
#        dd=nufft_ob(cc,tr.cuda(),smap=smapT[0].unsqueeze(0))
#        ee1=adjnufft_ob(dd*dcf.cuda(),tr.cuda(),smap=smapT[0].unsqueeze(0))

        def AtA_UV(x,vt):
            ktb=nufft_ob(x,ktrajT.cuda())
            ktb=torch.reshape(ktb,(nbasis,nch,2,NF,nintl*nx))
            ktb=ktb.permute(3,0,1,2,4)
            ktb=torch.reshape(ktb,(NF,nbasis,nch*2*nintl*nx))
            
            tmp1=torch.zeros((NF,nch*2*nintl*nx)).to('cuda:0')
            
            for i in range(nbasis):
                tmp1=tmp1+vt[i,]@ktb[:,i,:]
                
            vt=torch.reshape(vt,(nbasis*NF,NF))
            tmp2=vt@tmp1
            tmp2=torch.reshape(tmp2,(nbasis,NF,nch,2,nintl*nx))
            tmp2=tmp2.permute(0,2,3,1,4)
            tmp2=torch.reshape(tmp2,(nbasis,nch,2,NF*nintl*nx))
            atb1=adjnufft_ob(tmp2,ktrajT.cuda())
            #x=torch.reshape(x,(nbasis,1*2*nx*nx))
            #reg=sbasis.T@x
            #reg=torch.reshape(reg,(nbasis,1,2,nx,nx))
            #atb1=atb1+reg
            return atb1
        
        
        #%% congugate gradient parameter and code to get the final images.
        #lam=1e-2
        cgIter=5
        cgTol=1e-15
        atb=atb.unsqueeze(1)
        recT=torch.zeros_like(atb)
        Vn=np.reshape(Vn,(nbasis,NF,NF))
        B=lambda x: AtA_UV(x,torch.tensor(Vn).cuda())
        sf.tic()
        recT=sf.pt_cg(B,atb,recT,cgIter,cgTol)
        sf.toc()




        
        #del kdaT,smap,adjnufft_ob
        #%%
        z=atb#.permute(1,0,2,3)
        z=torch.reshape(z,(1,2*nbasis,nx,nx))
        del atb
        #z = torch.randn((1,60,512,512),device=gpu, dtype=dtype)
        z = Variable(z,requires_grad=False)
        z1=torch.reshape(torch.tensor(V),(1,1,nbasis,NF))
        z1 = Variable(z1,requires_grad=False)
        #u1=torch.zeros((30,2,512,512),device=gpu, dtype=dtype)
        #u1 = Variable(u1,requires_grad=True)
        #u1=u1.unsqueeze(1)
        v1=torch.tensor(V).cuda()
        
        optimizer=torch.optim.AdamW([{'params':G.parameters(),'lr':1e-3}])

        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=6, verbose=True, min_lr=5e-5)
        #%%    
        torch.cuda.empty_cache()
        
        loss=0
#        batch_sz=180
#        bb_sz=int(NF/batch_sz)
        #smapT=smapT[0].unsqueeze(0)
        #ktrajT=ktrajT[0].unsqueeze(0)
        #nufft_ob = KbNufft(im_size=im_size).to(dtype)
        #nufft_ob=nufft_ob.to(gpu)
        nufft_ob = MriSenseNufft(im_size=im_size,smap=smapT,norm='ortho').to(dtype)
        nufft_ob=nufft_ob.to(gpu)
        #del smapT
        #ktrajT=ktrajT.to(gpu)
        smapT=smapT.cpu()
        ktrajT=ktrajT.to(cpu)
        df=torch.reshape(dcf[0,0,0],(900,nintl*nx))
        df=df.unsqueeze(1).unsqueeze(1)
        df=df.repeat(1,3,2,1)
        df1=torch.reshape(df,(NF,3*2*nintl*nx))
        
#        df=df.permute(1,0,2)
#        df=torch.reshape(df,(900,1,2*nintl*nx))
#        df1=df.repeat(1,3,1)
#        df1=torch.reshape(df1,(900,3*10240))
        #xx=torch.zeros((nbasis,nch,2,NF*nx*nintl)).to(gpu)
        #xx=torch.cuda.FloatTensor(nbasis,nch,2,NF*nx*nintl).fill_(0)
        optimizer=torch.optim.SGD([{'params':u1,'lr':1e-2,'momentum':0.9}])
        sf.tic()
        for it in range(10):    
            for bat in range(1):
                #v1=GV(z1.cuda()).squeeze(0).squeeze(0)
                u1=G(z.cuda())
                #u2=u1
                u1=torch.reshape(u1,(nbasis,2,nx,nx))
                #u1=u1.permute(1,0,2,3)
                u1=u1.unsqueeze(1)
                #xx=nufft_ob(u1,ktrajT.cuda())
                yy=torch.cuda.FloatTensor(NF,nch*2*nx*nintl).fill_(0)
                for i in range(nbasis):
                    xx=nufft_ob(u1[i].unsqueeze(0),ktrajT[i].unsqueeze(0).cuda(),smap=smapT[i].unsqueeze(0).cuda())
                #xx=for_nufft(nufft_ob,u1,smapT,ktrajT)  
                    xx=torch.reshape(xx,(1,nch,2,NF,nx*nintl))
                    xx=xx.permute(0,3,1,2,4)
                    xx=torch.reshape(xx,(NF,nch*2*nx*nintl))
                #yy=torch.zeros((nbasis,NF,nch*2*nx*nintl)).to(gpu)
                #yy=torch.cuda.FloatTensor(NF,nch*2*nx*nintl).fill_(0)
                #for i in range(nbasis):
                    yy=yy+torch.diag(v1[i])@xx
                #yy=torch.sum(yy,dim=0)
                #yy=torch.reshape(yy,(NF,nch,2,nx*nintl))
                #del u1,v1,xx
    #            l1_reg = 0.
    #            for param in G.parameters():
    #                l1_reg += param.abs().sum()
                #loss = criterion(out, target) + l1_regularization
                err=(100*kdataT.cuda()-yy).pow(2)
                loss=(df1.cuda()*err).sum()#+0.1*(z.cpu()-u2.cpu()).pow(2).sum()+0.1*(z1[0,0].cpu()-v2.cpu()).pow(2).sum()
                #del yy
                #loss=loss.cuda()
                #if ep1%5==0:
                print(fl,loss.item())
                #torch.cuda.empty_cache()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step(loss.detach())
                torch.cuda.empty_cache()
    #    if ep1%20==0:
    #        torch.save(u1,'res_'+str(ep1)+'.pt')
        sf.toc()
    if ep1%2==0:
        wtsFname1='wts-10U'+str(ep1+1)+'.pt'
        torch.save(G.state_dict(),wtsFname1) 
        wtsFname2='wts-10V'+str(ep1+1)+'.pt'
        torch.save(GV.state_dict(),wtsFname2)      
#%%
#V=V.detach().cpu().numpy()
#
Vx=v1.detach().cpu().numpy()
Vz=V#V.detach().cpu().numpy()

for i in range(29,20,-1):
    plt.figure(i)
    plt.plot(Vz[i])
    plt.pause(0.1)
    plt.plot(Vx[i])
    #plt.pause()       
    plt.show() 
    
    
rec = np.squeeze(u1.detach().cpu().numpy())
#rec=np.reshape(rec,(nbasis,nx,nx,2))
rec = rec[:,0] + 1j*rec[:,1]
        
plt.figure(2)
for i in range(30):
    plt.imshow((np.squeeze(np.abs(rec[i,:,:]))),cmap='gray')       
    plt.show() 
    plt.pause(1)
#    
rec=np.reshape(rec,(nbasis,nx*nx))
rec1=rec.T@V[:,0:NF]
rec1=np.reshape(rec1,(nx,nx,NF))
#rec1=np.fft.fftshift(np.fft.fftshift(rec1,0),1)
#plt.figure(1,[15,15])
for i in range(100):
    plt.imshow((np.squeeze(np.abs(rec1[:,:,i]))),cmap='gray')       
    plt.show() 
    plt.pause(0.01)
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

rec=np.squeeze(atb.cpu().numpy())
rec=rec[:,0]+rec[:,1]*1j
