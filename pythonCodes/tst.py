"""
Created on Mon Mar 14, 2020

@author: haggarwal
"""

import os,time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import miscTorch as sf
import viewShots as vs
import readBrain as rd
from model import UnetClass

torch.cuda.empty_cache()
gpu=torch.device('cuda')

#%% Test on a single subject
nFrom=50
nImg=70
acc=4
sigma=.03

directory='24Mar_060743pm_4acc_360img_100ep_fix_with_noise'
chkPoint='100'

#%% get the dataset
tstDs=rd.DataGenFast('tst',nFrom,nImg,acc,sigma,False)
nImg=len(tstDs)
loader=DataLoader(tstDs,batch_size=1,shuffle=False,num_workers=8)
#%% create model and load the weights
unet=UnetClass()
modelDir='PTmodels/'+directory+'/wts-'+str(chkPoint)+'.pt'
unet.load_state_dict(torch.load(modelDir))
unet.eval()
unet.to(gpu)
#%% Do the testing

tstOrg=[]
tstAtb=[]
tstRec=[]

with torch.no_grad():
    for data in loader:
        slcOrg,slcAtb=data[0].to(gpu),data[1].to(gpu)
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

vs.viewer3(normOrg,normAtb,normRec,psnrAtb,psnrRec)

plt.show()