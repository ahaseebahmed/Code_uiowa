"""
Created on Mar 12, 2020
Train Unet only
@author: haggarwal
"""

import os,time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from shutil import copyfile
import distro

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


import miscTorch as sf
import viewShots as vs
import readBrain as rd
from model import UnetClass

torch.cuda.empty_cache()
gpu=torch.device('cuda')

#%%
nFrom=0
nImg=360
acc=4
sigma=.03

batchSz=1
epochs=100
savePeriod=50

#%% get the dataset
trnDs=rd.DataGenFast('trn',nFrom,nImg,acc,sigma,True)
#trnDs=rd.DataGen('trn',nFrom,nImg,acc,sigma,True)
nImg=len(trnDs)
loader=DataLoader(trnDs,batch_size=batchSz,shuffle=True,
                  num_workers=8,pin_memory=True)
#dataiter=iter(loader)
#org,atb=next(dataiter)
#o1=sf.pt_abs(sf.pt_f2b(org)).numpy()
#%%Generate a meaningful filename to save the trainined models for testing
print ('*************************************************')
start_time=time.time()
saveDir='PTmodels/'
cwd=os.getcwd()
directory=saveDir+datetime.now().strftime("%d%b_%I%M%S%P_")+ \
 str(acc)+'acc_'+  str(nImg)+'img_'+str(epochs)+'ep_' +'fix_with_noise'

if not os.path.exists(directory):
    os.makedirs(directory)
sessFileName= directory+'/model'
#backup the code
copyfile(cwd+'/trn.py',cwd+'/'+directory +'/trn.py')
copyfile(cwd+'/readBrain.py',cwd+'/'+directory +'/readBrain.py')
copyfile(cwd+'/miscTorch.py',cwd+'/'+directory +'/miscTorch.py')
copyfile(cwd+'/model.py',cwd+'/'+directory +'/model.py')

#%% creating the training model
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
writer.add_graph(unet,torch.randn(1,2,256,232).to(gpu))
for ep in range(epochs):
    epLoss = 0.0
    epStart=time.time()
    for i, data in enumerate(loader):
        org,atb = data[0].to(gpu), data[1].to(gpu)
        pred = unet(atb)
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
print ('Trianing completed in minutes ', round((end_time - start_time) / 60,2))
print ('training completed on', datetime.now().strftime("%d-%b-%Y at %I:%M %P"))
print ('*************************************************')

