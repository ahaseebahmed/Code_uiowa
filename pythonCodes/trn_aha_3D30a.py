"""
Created on Mar 12, 2020
Train Unet only
@author: haggarwal
"""

import os,time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import scipy.io as sio
import h5py as h5
import random

import readDataRad as rd
from spatial_model6 import SmallModel

torch.cuda.empty_cache()
gpu=torch.device('cuda')

#%%
#nFrom=0
#nImg=360
#acc=4
#sigma=.03
#
#batchSz=1
#epochs=100
savePeriod=2

epochs=500
N=512
NF=900
nc=2
nn=0
nx=270
patch_size=32
strde=32
noise_level=[0.07]
n_select=30
#%%Generate a meaningful filename to save the trainined models for testing
print ('*************************************************')
start_time=time.time()
saveDir='SPTmodels/'
cwd=os.getcwd()
directory=saveDir+datetime.now().strftime("%d%b_%I%M%S%P_")+ \
 str(epochs)+'ep_' +'18Jun'

if not os.path.exists(directory):
    os.makedirs(directory)
sessFileName= directory+'/model1c'
#backup the code
#copyfile(cwd+'/trn.py',cwd+'/'+directory +'/trn.py')
#copyfile(cwd+'/readBrain.py',cwd+'/'+directory +'/readBrain.py')
#copyfile(cwd+'/miscTorch.py',cwd+'/'+directory +'/miscTorch.py')
#copyfile(cwd+'/model.py',cwd+'/'+directory +'/model.py')

#%% creating the training model
unet=SmallModel()
#modelDir1='SPTmodels/15Jun_093441pm_50ep_13Jun/wts-'+str(4)+'.pt'
#unet.load_state_dict(torch.load(modelDir1))
unet=unet.to(gpu)

#criterion = nn.MSELoss()
optimizer=torch.optim.Adam(unet.parameters(),lr=1e-3)
def lossFun(pred,org):
    loss=torch.mean(torch.abs(pred-org))
    return loss
#%% training code
print ('training started on', datetime.now().strftime("%d-%b-%Y at %I:%M %P"))
start_time=time.time()
torch.save(unet.state_dict(), directory+'/wts-0.pt')
#writer = SummaryWriter(log_dir=directory+'/')
#writer.add_graph(unet,torch.randn(1,2,512,512,30).to(gpu))
trnFiles=os.listdir('./../Codes/TensorFlow_SToRM/Data/')
sz=1#len(trnFiles)
data2=np.zeros((sz,1,n_select,N,N)).astype(np.complex64)
rndm=random.sample(range(sz),sz)
rndm=int(74)
for fl in range(sz):
    matfile = sio.loadmat('./../Codes/TensorFlow_SToRM/Data/'+trnFiles[74])
    #matfile = sio.loadmat('./../../../localscratch/Users/ahhmed/'+trnFiles[rndm[fl]])
    data0 = matfile['D']
    data0=np.transpose(data0,(1,0))
    data1 = matfile['U1']
    data0=data0[:,0:1*n_select]
    data=np.matmul(data1,data0)
    data=np.transpose(data,(1,0))
    data=np.reshape(data,(1*n_select,N,N))
    #data=data[0:6*n_select,]
    data=np.fft.fftshift(np.fft.fftshift(data,1),2) 
#        data=data[:,110:380,110:380]
    data=data.astype(np.complex64)
    data=(data)/(np.max(np.abs(data)))
    data=np.reshape(data,(1,n_select,N,N))
    data2[fl]=data
#data2=np.transpose(data2,(0,1,3,4,2))
data2=np.reshape(data2,(sz*1,n_select,N,N))
inp=np.stack((np.real(data2),np.imag(data2)),axis=1)

Inp=(inp)
trnOrg=(inp)
#        trnInp=np.transpose(trnInp,(1,2,3,0))
#        trnOrg=np.transpose(trnOrg,(1,2,3,0))
    #trnInp=np.expand_dims(trnInp,axis=0)
    #trnOrg=np.expand_dims(trnOrg,axis=0)
#%%
for ep in range(epochs):
    epLoss = 0.0
    epStart=time.time()
    for nn in range(len(noise_level)):
        noise1=np.random.normal(0,noise_level[nn],(inp.shape[0],inp.shape[1],inp.shape[2],inp.shape[3],inp.shape[4]))
        #noise1=torch.normal(0,noise_level[nn],size=(inp.shape[0],inp.shape[1],inp.shape[2],inp.shape[3],inp.shape[4]))
        trnInp=Inp + noise1
        #trnDs=TensorDataset(trnInp,trnOrg)
        trnDs=rd.DataGen(trnInp,trnOrg)
        loader=DataLoader(trnDs,batch_size=1,shuffle=True,num_workers=8,pin_memory=True)
        for i, (inp,org) in enumerate(loader):
            org,inp = org.to(gpu,dtype=torch.float), inp.to(gpu,dtype=torch.float)
            pred = unet(inp)
            loss = lossFun(pred, org)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epLoss += loss.item()
    epEnd=time.time()
    epTime=(epEnd - epStart) / 60
    print('Epoch:%d, loss: %.3f, time: %.2f min' % (ep + 1, epLoss,epTime))
#writer.add_scalar('training_loss', epLoss,global_step=ep+1)
    if ((ep+1) % savePeriod==0) or ((ep+1)==epochs) :
        wtsFname=directory+'/wts-'+str(ep+1)+'.pt'
        torch.save(unet.state_dict(), wtsFname)
#writer.close()


end_time = time.time()
rec1= np.squeeze(pred.detach().cpu().numpy())
rec2=rec1[0]+1j*rec1[1]
plt.imshow(np.abs(rec2[:,:,0]))
print ('Trianing completed in minutes ', round((end_time - start_time) / 60,2))
print ('training completed on', datetime.now().strftime("%d-%b-%Y at %I:%M %P"))
print ('*************************************************')

