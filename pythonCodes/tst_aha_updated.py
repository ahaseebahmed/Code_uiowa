"""
@author: abdul haseeb
"""

import os,time
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import scipy.io as sio
from spatial_model6 import SmallModel

torch.cuda.empty_cache()
gpu=torch.device('cuda')

#%% Test on a single subject
nn=0
NF=900
N=512
directory=''
chkPoint=''
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
data=data[0:10,:,:]
NF=10

data=np.fft.fftshift(np.fft.fftshift(data,1),2) 
data=data.astype(np.complex64)
data=(data)/(np.max(np.abs(data)))
inp=np.stack((np.real(data),np.imag(data)),axis=1)
trnInp=np.zeros((NF*len(noise_level),inp.shape[1],inp.shape[2],inp.shape[3]))
noise1=np.random.normal(0,noise_level,(inp.shape[0],inp.shape[1],inp.shape[2],inp.shape[3]))
trnInp[NF*nn:NF*(nn+1),:,:,:]=inp + noise1
trnOrg=np.tile(inp,(len(noise_level),1,1,1))
trnDs=TensorDataset(torch.Tensor(trnInp),torch.Tensor(trnOrg))
loader=DataLoader(trnDs,batch_size=1,shuffle=False,num_workers=8,pin_memory=True)

#%% create model and load the weights
model=SmallModel()
model=model.to(gpu)
modelDir='SPTmodels/'+directory+'/wts-'+str(chkPoint)+'.pt'
model.load_state_dict(torch.load(modelDir))
model.eval()
#%% Do the testing
tstRec=[]

with torch.no_grad():
    for data in loader:
        slcOrg,slcAtb=data[0].to(gpu,dtype=torch.float),data[1].to(gpu,dtype=torch.float)
        slcRec=model(slcAtb)
        tstRec.append(slcRec)


rec=torch.cat(tstRec)
rec= np.squeeze(rec.cpu().numpy())
rec = rec[:,0] + 1j*rec[:,1]

#%%
for i in range(100):
    plt.imshow((np.squeeze(np.abs(rec[:,:,i]))),cmap='gray')       
    plt.show() 
    plt.pause(0.05)
