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
from torch.utils.data import TensorDataset


import scipy.io as sio
import miscTorch as sf
#import viewShots as vs
import readDataRad as rd
from spatial_model6 import SmallModel

torch.cuda.empty_cache()
gpu=torch.device('cuda')

#%% Test on a single subject
nFrom=50
nImg=70
acc=4
sigma=.03
NF=900
N=512
directory='21Jun_110930pm_100ep_21Jun'
chkPoint='88'
noise_level=[0.03]
#%%
trnFiles=os.listdir('./../Codes/TensorFlow_SToRM/Data/')
matfile = sio.loadmat('./../Codes/TensorFlow_SToRM/Data/'+trnFiles[74])
data0 = matfile['D']
data0=np.transpose(data0,(1,0))
data1 = matfile['U1']
data=np.matmul(data1,data0)
data=np.transpose(data,(1,0))
data=np.reshape(data,(NF,N,N))
data=data[0:60,:,:]
NF=60

data=np.fft.fftshift(np.fft.fftshift(data,1),2) 
#data=data[:,110:380,110:380]
data=data.astype(np.complex64)
data=(data)/(np.max(np.abs(data)))
inp=np.stack((np.real(data),np.imag(data)),axis=1)
inp=np.reshape(inp,(2,30,2,512,512))
inp=np.transpose(inp,(0,2,3,4,1))
trnInp=np.zeros((2*len(noise_level),inp.shape[1],inp.shape[2],inp.shape[3],inp.shape[4]))
nn=0
noise1=np.random.normal(0,noise_level,(inp.shape[0],inp.shape[1],inp.shape[2],inp.shape[3],inp.shape[4]))
#trnInp[2*nn:2*(nn+1),:,:,:,:]=inp + noise1
trnInp=inp + noise1

#trnOrg=np.tile(inp,(len(noise_level),1,1,1,1))
trnOrg=inp
#trnInp=torch.tensor(trnInp)
#trnOrg=torch.tensor(trnOrg)

trnDs=rd.DataGen(trnInp,trnOrg)
loader=DataLoader(trnDs,batch_size=1,shuffle=False,num_workers=8)
#%% get the dataset
#tstDs=rd.DataGenFast('tst',nFrom,nImg,acc,sigma,False)
#nImg=len(tstDs)
#loader=DataLoader(tstDs,batch_size=1,shuffle=False,num_workers=8)
#%% create model and load the weights
unet=SmallModel()
modelDir='SPTmodels/'+directory+'/wts-'+str(chkPoint)+'.pt'
unet.load_state_dict(torch.load(modelDir))
unet.eval()
unet.to(gpu)
#%% Do the testing

tstOrg=[]
tstAtb=[]
tstRec=[]

with torch.no_grad():
    for dat in loader:
        slcOrg,slcAtb=dat[1].to(gpu,dtype=torch.float),dat[0].to(gpu,dtype=torch.float)
        slcRec=unet(slcAtb)
        tstOrg.append(slcOrg)
        tstAtb.append(slcAtb)
        tstRec.append(slcRec)


#%%
rec=torch.cat(tstRec)
rec= np.squeeze(rec.cpu().numpy())
rec = rec[:,0] + 1j*rec[:,1]

rec=np.transpose(rec,(0,3,1,2))
rec=np.reshape(rec,(2*30,N,N))


rec1=torch.cat(tstOrg)
rec1= np.squeeze(rec1.cpu().numpy())
rec1 = rec1[:,0] + 1j*rec1[:,1]

rec1=np.transpose(rec1,(0,3,1,2))
rec1=np.reshape(rec1,(2*30,N,N))

rec2=np.abs(rec[:,200:400,200:400])
rec3=np.abs(rec1[:,200:400,200:400])
rc=np.concatenate((rec2,rec3),axis=2)
#%%
for i in range(60):
    plt.imshow((np.squeeze(np.abs(rc[i,:,:]))),cmap='gray')       
    plt.show() 
    plt.pause(0.1)

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