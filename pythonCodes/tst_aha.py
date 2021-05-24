"""
Created on Mon Mar 14, 2020

@author: haggarwal
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import readDataRad as rd

import scipy.io as sio
import miscTorch as sf
#import viewShots as vs
import readDataRad as rd
from spatial_model6_2D import SmallModel
#from model import UnetClass

torch.cuda.empty_cache()
gpu=torch.device('cuda')

#%% Test on a single subject
nFrom=50
nImg=70
acc=4
sigma=.03
NF=900
N=512
directory='19Jun_083427pm_400ep_19Jun'#'28Apr_030506pm_41ep_noise_1'
chkPoint='100'#'41'
noise_level=[0.07]
#%%
trnFiles=os.listdir('./../Codes/TensorFlow_SToRM/Data/')
matfile = sio.loadmat('./../Codes/TensorFlow_SToRM/Data/'+trnFiles[74])
data0 = matfile['D']
data0=np.transpose(data0,(1,0))
data1 = matfile['U1']
data=np.matmul(data1,data0)
data=np.transpose(data,(1,0))
data=np.reshape(data,(NF,N,N))
data=data[0:10,:,:]
NF=10

data=np.fft.fftshift(np.fft.fftshift(data,1),2) 
#data=data[:,110:380,110:380]
data=data.astype(np.complex64)
data=(data)/(np.max(np.abs(data)))
inp=np.stack((np.real(data),np.imag(data)),axis=1)
trnInp=np.zeros((NF*len(noise_level),inp.shape[1],inp.shape[2],inp.shape[3]))
nn=0
noise1=np.random.normal(0,noise_level,(inp.shape[0],inp.shape[1],inp.shape[2],inp.shape[3]))
trnInp[NF*nn:NF*(nn+1),:,:,:]=inp + noise1
trnOrg=np.tile(inp,(len(noise_level),1,1,1))
trnDs=rd.DataGen(trnInp,trnOrg)
nImg=len(trnDs)
loader=DataLoader(trnDs,batch_size=10,shuffle=False,num_workers=8)
#%% get the dataset
#tstDs=rd.DataGenFast('tst',nFrom,nImg,acc,sigma,False)
#nImg=len(tstDs)
#loader=DataLoader(tstDs,batch_size=1,shuffle=False,num_workers=8)
#%% create model and load the weights
unet=SmallModel()
#unet=UnetClass()

modelDir='SPTmodels/'+directory+'/wts-'+str(chkPoint)+'.pt'
unet.load_state_dict(torch.load(modelDir))
unet.eval()
unet.to(gpu)
#%% Do the testing

tstOrg=[]
tstAtb=[]
tstRec=[]

with torch.no_grad():
    for data in loader:
        slcOrg,slcAtb=data[1].to(gpu,dtype=torch.float),data[0].to(gpu,dtype=torch.float)
        slcRec=unet(slcAtb)
        tstOrg.append(slcOrg)
        tstAtb.append(slcAtb)
        tstRec.append(slcRec)

fn=lambda x: sf.pt_f2b(torch.cat(x)).cpu().numpy()

tstOrg=fn(tstOrg)
tstAtb=fn(tstAtb)
tstRec=fn(tstRec)

#%%
fn= lambda x: sf.normalize01(np.abs(sf.r2c(x)))

normOrg=fn(tstOrg)
normAtb=fn(tstAtb)
normRec=fn(tstRec)

#%%
psnrAtb=sf.myPSNR(normOrg,normAtb)
ssimAtb=sf.mySSIM(normOrg,normAtb)

psnrRec=sf.myPSNR(normOrg,normRec)
ssimRec=sf.mySSIM(normOrg,normRec)

print ('  ' + 'Noisy ' + 'Rec')
print ('  {0:.2f} {1:.2f}'.format(psnrAtb.mean(),psnrRec.mean()))
print ('  {0:.2f} {1:.2f}'.format(ssimAtb.mean(),ssimRec.mean()))
#%%

# fn= lambda x: np.rot90( sf.crop(x,(320,320)), k=-2,axes=(-1,-2))
# normOrg=fn(normOrg)
# normAtb=fn(normAtb)
# normRec=fn(normRec)

#vs.viewer3(normOrg,normAtb,normRec,psnrAtb,psnrRec)
#
#plt.show()