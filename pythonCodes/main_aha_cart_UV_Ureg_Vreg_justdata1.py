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

#from unetmodel import UnetClass
from unetmodel2 import UnetClass

from dn_modelV2 import SmallModel

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


NF=200#100#900 
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
#%% make them pytorch tensors


VT=torch.load('VT.pt')
#W=torch.load('W200.pt')
#VT=torch.load('VT_200.pt')

recT=torch.load('stm200L200W.pt')
#recT=torch.load('stm200LW.pt')

#recT=torch.load('stm900U.pt')
atbT=torch.load('atb200.pt')

#%% take them to gpu
#B=B.to(gpu)
#atbT=atbT.to(gpu)


VT=VT.to(gpu)
recT=recT.to(gpu)
#%%
def maskV(msk,Vv):
    bsz=msk.size(0)
    res=torch.zeros((bsz,nbasis,nx*nx))
    res=res.to(gpu)
    for k in range(bsz):
        tmp3=msk[k].repeat(nbasis,1)#*tmp2
        res[k]=torch.diag(Vv[k])@tmp3.to(torch.float32)
    return res

#%%
wtsFname1='wts-3U'+str(62+1)+'.pt'
#wtsFname1='wts-2U'+str(60+1)+'.pt'

G=UnetClass().to(gpu)
G.load_state_dict(torch.load(wtsFname1))

#atbT=atbT*W
z=atbT.permute(3,0,1,2)
z=torch.reshape(z,(1,2*nbasis,512,512)).to(gpu)
#z = torch.randn((1,60,512,512),device=gpu, dtype=dtype)
z = Variable(z,requires_grad=True)


wtsFname2='wts-3V'+str(62+1)+'.pt'
#wtsFname2='wts-2V'+str(60+1)+'.pt'

GV=SmallModel().to(gpu)
GV.load_state_dict(torch.load(wtsFname2))

#z1 = torch.randn((1,30,900),device=gpu, dtype=dtype)
z1=torch.reshape(VT,(1,1,nbasis,NF))
z1 = Variable(z1,requires_grad=False)

#optimizer=torch.optim.SGD([{'params':G.parameters(),'lr':5e-3,'momentum':0.9}])
optimizer1=torch.optim.AdamW([{'params':z,'lr':1e-3},{'params':G.parameters(),'lr':1e-3}])
scheduler = ReduceLROnPlateau(optimizer1, mode='min', factor=0.7, patience=6, verbose=True, min_lr=5e-5)

#z1=z
sf.tic()
for ep1 in range(5):
    for bat in range(1):
        u1=G(z)
        #U=U.permute(1,2,3,0)
        u1=torch.reshape(u1,(2,nbasis,nx,nx))
        u1=u1.permute(1,2,3,0)
        u1=torch.reshape(u1,(nbasis,nx,nx,2))
        loss=abs(u1-recT).pow(2).sum()
        if ep1%10==0:
            print(ep1,loss.item())
        
        optimizer1.zero_grad()
        loss.backward()
        optimizer1.step()
        scheduler.step(loss.detach())
sf.toc()

optimizer2=torch.optim.AdamW([{'params':GV.parameters(),'lr':1e-4}])
scheduler = ReduceLROnPlateau(optimizer2, mode='min', factor=0.7, patience=6, verbose=True, min_lr=5e-5)

sf.tic()
for ep1 in range(500):
    for bat in range(1):
        v1=GV(z1)
        #v1=v1.permute(1,0)
        loss=abs(v1[0,0]-VT).pow(2).sum()
        if ep1%10==0:
            print(ep1,loss.item())
        
        optimizer2.zero_grad()
        loss.backward()
        optimizer2.step()
        scheduler.step(loss.detach())
sf.toc()        
#torch.save(G.state_dict(),'tempU.pt')
#del recT

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
        return tmp5.cpu()

#%%

torch.cuda.empty_cache()

csmT=torch.load('csm.pt')
csmT=csmT.to(gpu)
B=torch.load('B.pt')
maskT=torch.load('maskT.pt').to(dtype=torch.float16)
#recT=torch.load('stm900U.pt')
dd=sio.loadmat('stm900o.mat')
org=np.asarray(dd['org']) 
org=org.transpose(2,0,1)
#B=torch.tensor(sf.c2r(kdata1))


#res=maskV(maskTst,v1.detach().cpu().numpy())
#maskT=torch.tensor(maskTst).to(dtype=torch.float32)



loss=0
#indx=torch.randint(0,NF,(NF,))
#indx=torch.tensor(indx)
#B=B[:,indx]
#maskT=maskT[indx,]
batch_sz=25
bb_sz=int(NF/batch_sz)
B=torch.reshape(B,(nch,int(NF/batch_sz),batch_sz,nx*nx*2))
maskT=torch.reshape(maskT,(int(NF/batch_sz),batch_sz,nx*nx))

F=AUV(csmT,nbasis,batch_sz)
maskT=maskT.to(gpu)

z = Variable(z,requires_grad=True)
optimizer=torch.optim.Adam([{'params':z,'lr':1e-4},{'params':G.parameters(),'lr':1e-4},{'params':GV.parameters(),'lr':1e-4}])
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=6, verbose=True, min_lr=5e-5)

pp=np.array([])
pp1=np.array([])
ppu=np.array([])
pp2=np.array([])


sf.tic()
for ep1 in range(120):
    #indx=torch.randint(0,bb_sz,(bb_sz,))
    for bat in range(bb_sz):
        v1=GV(z1)
        v1=v1.squeeze(0).squeeze(0)
        v2=v1.permute(1,0)
        v2=torch.reshape(v2,(int(NF/batch_sz),batch_sz,nbasis))
        #mask_b=maskT[bat]
        m_res=maskV(maskT[bat],v2[bat])
        m_res=m_res.unsqueeze(3).repeat(1,1,1,2)
        m_res=torch.reshape(m_res,(batch_sz,nbasis,nx*nx*2))
        u1=G(z)
        u1=torch.reshape(u1,(2,nbasis,nx,nx))
        u1=u1.permute(1,2,3,0)
        u1=torch.reshape(u1,(nbasis,nx*nx*2))
        b_est=F(u1,m_res.to(gpu))
        l1_reg = 0.
        l2_reg = 0.
        for param in G.parameters():
            l1_reg += param.abs().sum()
            
        for param in GV.parameters():
            l2_reg += param.abs().sum()
        #loss = criterion(out, target) + l1_regularization
        loss=(b_est-B[:,bat]).pow(2).sum()+0.0001*l1_reg.cpu() + 0.1*l2_reg.cpu()#+(sT@W*u1).pow(2).sum()
#        if ep1%5==0:
#            print(ep1,loss.item())
        loss=loss.cuda()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(loss.detach())
        #orgD=VT.cpu().T@recT.view(nbasis,nx*nx*2).cpu()
        #orgD=orgD/torch.max(orgD)
        recD=v1.detach().cpu().T@u1.cpu().detach()
        #recD=recD/torch.max(recD)
        #org=orgD.view(NF,nx,nx,2).cpu().numpy()
        recon=recD.view(NF,nx,nx,2).cpu().numpy() 
        #org1=org[:,190:300,190:300,0]+org[:,190:300,190:300,1]*1j
        #recon1=recon[:,190:300,190:300,0]+recon[:,190:300,190:300,1]*1j
        #org1=org[:,100:400,100:400,0]+org[:,100:400,100:400,1]*1j
        recon1=recon[:,100:400,100:400,0]+recon[:,100:400,100:400,1]*1j
        org1=np.abs(org[:,100:400,100:400])
        recon1=np.abs(recon1)
        recon1=sf.scale(org1,recon1)
        psnr=np.mean(sf.myPSNR1(org1,recon1))
        xx=recT.cpu().numpy()
        yy=u1.view(nbasis,nx,nx,2).detach().cpu().numpy()
        x1=xx[:,:,:,0]+xx[:,:,:,1]*1j
        x2=yy[:,:,:,0]+yy[:,:,:,1]*1j
        x1=np.reshape(x1,(nbasis,nx,nx))
        x2=np.reshape(x2,(nbasis,nx,nx))
        psnr1=np.mean(sf.myPSNR(np.abs(x1),np.abs(x2)))
        del recD
        #p=np.mean(sf.myPSNR(recT.detach().cpu().numpy(),torch.reshape(u1,(nbasis,nx,nx,2)).detach().cpu().numpy()))
        print(ep1,'snr',psnr,psnr1)
        pp=np.append(pp,psnr)
        ppu=np.append(ppu,psnr1)
    #print(ep1)
#    p=np.mean(pp)
#    pu=np.mean(ppu)
#    pp1=np.append(pp1,p)
#    pp2=np.append(pp2,pu)
#    print(ep1,'snr',p,pu)
#    pp=np.array([])
#    ppu=np.array([])
    
    
#    if ep1<=20 or ep1== 39 or ep1 ==199:
#        torch.save(u1.detach().cpu(),'dpu6_'+str(ep1)+'.pt')
#        torch.save(v1.detach().cpu(),'dpv6_'+str(ep1)+'.pt')
sf.toc()
#torch.save(pp,'pdp5_snr.pt')
#torch.save(pp1,'pdp5_snrU.pt')
#del v1,v2,m_res,u1, 
#del G, GV, maskT, csmT, maskT1, VT, sT,Nv
#        
#%%
V=v1.detach().cpu().numpy()
plt.figure(2)
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
    plt.imshow((np.squeeze(np.abs(rec1[100:400,100:400,i]))),cmap='gray')       
    plt.show() 
    plt.pause(0.05)

nx=512
nx1=80
nbasis=30
NF=200
u1=torch.load('pdpu6_0.pt')
v1=torch.load('pdpv6_0.pt')

V=v1.detach().cpu().numpy()
rec = np.squeeze(u1.detach().cpu().numpy())
rec=np.reshape(rec,(nbasis,nx,nx,2))
rec = rec[:,210:290,210:290,0] + 1j*rec[:,210:290,210:290,1]

rec=np.reshape(rec,(nbasis,nx1*nx1))
rec1=rec.T@V[:,0:NF]
rec1=np.reshape(rec1,(nx1,nx1,NF))

u2=torch.load('pdpu6_20.pt')
v2=torch.load('pdpv6_20.pt')
V=v2.detach().cpu().numpy()
rec = np.squeeze(u2.detach().cpu().numpy())
rec=np.reshape(rec,(nbasis,nx,nx,2))
rec = rec[:,210:290,210:290,0] + 1j*rec[:,210:290,210:290,1]

rec=np.reshape(rec,(nbasis,nx1*nx1))
rec2=rec.T@V[:,0:NF]
rec2=np.reshape(rec2,(nx1,nx1,NF))

u3=torch.load('pdpu6_39.pt')
v3=torch.load('pdpv6_39.pt')
V=v3.detach().cpu().numpy()
rec = np.squeeze(u3.detach().cpu().numpy())
rec=np.reshape(rec,(nbasis,nx,nx,2))
rec = rec[:,210:290,210:290,0] + 1j*rec[:,210:290,210:290,1]

rec=np.reshape(rec,(nbasis,nx1*nx1))
rec3=rec.T@V[:,0:NF]
rec3=np.reshape(rec3,(nx1,nx1,NF))



for i in range(100):
    plt.imshow((np.squeeze(np.abs(org[i,:,:]))),cmap='gray')       
    plt.show() 
    plt.pause(0.05)


sio.savemat('pdp6_wol1.mat',{"rec1":rec1,"rec2":rec2,"rec3":rec3,"pp":pp})
sio.savemat('deblur7_U_0.0001Vl1_0.1.mat',{"pp":pp})
#sio.savemat('res200_3oct.mat',{"rec":rec1}) 
 
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