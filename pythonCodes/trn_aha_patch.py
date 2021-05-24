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
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import scipy.io as sio
import h5py as h5

import readDataRad as rd
#from spatial_model6_2D import SmallModel
from model import UnetClass

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
savePeriod=50

epochs=400
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
 str(epochs)+'ep_' +'19Jun'

if not os.path.exists(directory):
    os.makedirs(directory)
sessFileName= directory+'/model1c'
#backup the code
#copyfile(cwd+'/trn.py',cwd+'/'+directory +'/trn.py')
#copyfile(cwd+'/readBrain.py',cwd+'/'+directory +'/readBrain.py')
#copyfile(cwd+'/miscTorch.py',cwd+'/'+directory +'/miscTorch.py')
#copyfile(cwd+'/model.py',cwd+'/'+directory +'/model.py')

#%% creating the training model
#unet=SmallModel()
unet=UnetClass()

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
writer = SummaryWriter(log_dir=directory+'/')
writer.add_graph(unet,torch.randn(1,2,512,512,30).to(gpu))
trnFiles=os.listdir('./../Codes/TensorFlow_SToRM/Data/')
for ep in range(epochs):
    epLoss = 0.0
    epStart=time.time()
    for fl in range(1):#range(len(trnFiles)):
        matfile = sio.loadmat('./../Codes/TensorFlow_SToRM/Data/'+trnFiles[74])
        data0 = matfile['D']
        data0=np.transpose(data0,(1,0))
        data1 = matfile['U1']
        data=np.matmul(data1,data0)
        data=np.transpose(data,(1,0))
        data=np.reshape(data,(NF,N,N))
        data=data[0:1*n_select,]
        data=np.fft.fftshift(np.fft.fftshift(data,1),2) 
#        data=data[:,110:380,110:380]
        data=data.astype(np.complex64)
        data=(data)/(np.max(np.abs(data)))
        
        inp=np.stack((np.real(data),np.imag(data)),axis=1)
        inp=np.reshape(inp,(n_select,2,N,N))
        #inp=np.transpose(inp,(0,2,3,4,1))
        #inp=np.expand_dims(inp,axis=0)

        trnInp=np.zeros((1*len(noise_level),inp.shape[1],inp.shape[2],inp.shape[3],inp.shape[4]))

        for nn in range(len(noise_level)):
            noise1=np.random.normal(0,noise_level[nn],(inp.shape[0],inp.shape[1],inp.shape[2],inp.shape[3],inp.shape[4]))
            trnInp[nn,:,:,:,:]=inp + noise1
        trnOrg=np.tile(inp,(len(noise_level),1,1,1,1))
#        trnInp=np.transpose(trnInp,(1,2,3,0))
#        trnOrg=np.transpose(trnOrg,(1,2,3,0))
        #trnInp=np.expand_dims(trnInp,axis=0)
        #trnOrg=np.expand_dims(trnOrg,axis=0)
        trnDs=rd.DataGen(trnInp,trnOrg)
        nImg=len(trnDs)
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
    writer.add_scalar('training_loss', epLoss,global_step=ep+1)

    if ((ep+1) % savePeriod==0) or ((ep+1)==epochs) :
        wtsFname=directory+'/wts-'+str(ep+1)+'.pt'
        torch.save(unet.state_dict(), wtsFname)
writer.close()


end_time = time.time()
rec1= np.squeeze(pred.detach().cpu().numpy())
rec2=rec1[0]+1j*rec1[1]
plt.imshow(np.abs(rec2[:,:,0]))
print ('Trianing completed in minutes ', round((end_time - start_time) / 60,2))
print ('training completed on', datetime.now().strftime("%d-%b-%Y at %I:%M %P"))
print ('*************************************************')

