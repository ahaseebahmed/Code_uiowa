"""
Created on Tue Jun 30 13:58:06 2020
@author: ahhmed
"""

import numpy as np
import os, time
import matplotlib.pyplot as plt
import itertools
import pickle
import imageio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import h5py as h5
from torchkbnufft.torchkbnufft import KbNufft
from torchkbnufft.torchkbnufft import AdjKbNufft
from torchkbnufft.torchkbnufft import MriSenseNufft
from torchkbnufft.torchkbnufft import AdjMriSenseNufft
from torchkbnufft.torchkbnufft.nufft.sparse_interp_mat import precomp_sparse_mats


directory='/Users/ahhmed/pytorch_sense/'
dtype = torch.float

# results saving folder
if not os.path.isdir('DMRI_result'):
    os.mkdir('DMRI_result')
if not os.path.isdir('DMRI_result/results'):
    os.mkdir('DMRI_result/results')

NF=200 
nx=340
nch=3
nkpts=2336
nintl=6

#%%

with h5.File(directory+'csm_se17.h5', 'r') as f:  
  # coil sensitivity maps
  csm=np.asarray(f['/csm/re'])+np.asarray(f['/csm/im'])*1j
  csm=csm.astype(np.complex64)
  #csm=np.transpose(csm,(0,2,1))
  #csmTrn=np.transpose(csm,[0,2,1])
  #csm=np.fft.fftshift(np.fft.fftshift(csm,1),2)
  csmTrn=csm[0:nch,:,:]
  ncoils=csmTrn.shape[0]
  del csm
  
with h5.File(directory+'kdata_se17.h5', 'r') as f:  
  # reading the RHS: AHb
  kdata=np.asarray(f['/kdata/re'])
  kdata=kdata+np.asarray(f['/kdata/im'])*1j
  kdata=kdata.astype(np.complex64)
  kdata=np.reshape(kdata,(nch,NF,1,nintl*nkpts))
  kdata=np.reshape(kdata,(nch,NF,1*nintl*nkpts))
  kdata=np.squeeze(kdata[0:nch,0:NF,:])
  kdata=np.transpose(kdata,(1,0,2))  

with h5.File(directory+'dcf.h5', 'r') as f:  
  # reading the RHS: AHb
  dcf=np.asarray(f['/dcf/re'])
  dcf=np.reshape(dcf,(NF,1*nintl*nkpts))
  dcf=np.tile(dcf,(3,2,1,1))
  dcf=np.transpose(dcf,(2,0,1,3))

with h5.File(directory+'ktraj_se17.h5', 'r') as f: 
  ktraj=np.asarray(f['/csm/re'])
  ktraj=ktraj+np.asarray(f['/csm/im'])*1j
  ktraj=ktraj.astype(np.complex64)
  ktraj=np.reshape(ktraj,(NF,1,nintl*nkpts))
  ktraj=np.reshape(ktraj,(NF,1*nintl*nkpts))
  ktraj=np.squeeze(np.transpose(ktraj[0:NF,:],[0,1]))*2*np.pi

#%%
im_size = csmTrn[0].shape

kdata = np.stack((np.real(kdata), np.imag(kdata)),axis=2)
# convert to tensor, unsqueeze batch and coil dimension
# output size: (1, 1, 2, ny, nx)
kdataT = torch.tensor(kdata).to(dtype)

dcfT = torch.tensor(dcf).to(dtype)

smap = np.stack((np.real(csmTrn), np.imag(csmTrn)), axis=1)
smap=np.tile(smap,(NF,1,1,1,1))
smapT = torch.tensor(smap).to(dtype)
# convert to tensor, unsqueeze batch dimension
# output size: (1, 2, klength)
ktraj = np.stack((np.real(ktraj), np.imag(ktraj)), axis=1)
ktrajT = torch.tensor(ktraj).to(dtype)

#%% take them to gpu
kdataT=kdataT.cuda()
smapT=smapT.cuda()
ktrajT=ktrajT.cuda()
dcfT=dcfT.cuda()

#%% generate atb (Initial reconstruction)
nuf_ob = KbNufft(im_size=im_size).to(dtype)
nuf_ob=nuf_ob.cuda()
adjnuf_ob = AdjKbNufft(im_size=im_size).to(dtype)
adjnuf_ob=adjnuf_ob.cuda()
real_mat, imag_mat = precomp_sparse_mats(ktrajT, adjnuf_ob)
interp_mats = {'real_interp_mats': real_mat, 'imag_interp_mats': imag_mat}

nufft_ob = MriSenseNufft(im_size=im_size, smap=smapT).to(dtype)
nufft_ob=nufft_ob.cuda()
adjnufft_ob = AdjMriSenseNufft(im_size=im_size, smap=smapT).to(dtype)
adjnufft_ob=adjnufft_ob.cuda()

At=lambda x: adjnufft_ob(x*dcfT,ktrajT,interp_mats)
A=lambda x: nufft_ob(x,ktrajT,interp_mats)

#inverse NUFFT transform -->Initial reconstruction
atbT=At(kdataT) #Output size: 200x2x340x340

#%% Visualization of the first five initial recon.
atbT1 = atbT.cpu().data.numpy()
# atbT1 = atbT.numpy()
atbT1 = atbT1.squeeze()
atbT1 = atbT1[:,0,:,:] + atbT1[:,1,:,:]*1j

fig = plt.figure(figsize=(20, 10))

for k in range(5):
    plt.subplot(1, 5, k+1)
    plt.imshow(abs(atbT1[k,:,:]), cmap='gray')
    plt.xticks([]);plt.yticks([])

plt.tight_layout()
plt.savefig('DMRI_result/ATb.png')
plt.close()
    
    
# kdataT_new=A(atbT)#NUFFT transform

#%%
def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

#Define the generative model
class generator(nn.Module):
    # initializers
    def __init__(self,siz_latent=100, d=128, out_channel=2):
        super(generator, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(siz_latent, d*4, 3, 1, 0) # 1x1xsiz_latent->3x3x512
        self.deconv1_bn = nn.BatchNorm2d(d*4)
        self.deconv2 = nn.ConvTranspose2d(d*4, d*2, 3, 1, 0) # 3x3x512->5x5x256
        self.deconv2_bn = nn.BatchNorm2d(d*2)
        self.deconv3 = nn.ConvTranspose2d(d*2, d, 4, 2, 1) # 5x5x256->10x10x128
        self.deconv3_bn = nn.BatchNorm2d(d)
        self.deconv4 = nn.ConvTranspose2d(d, d, 4, 2, 1) # 10x10x128->20x20x128
        self.deconv4_bn = nn.BatchNorm2d(d)
        self.deconv5 = nn.ConvTranspose2d(d, int(d/2), 3, 2, 0) # 20x20x128->41x41x64
        self.deconv5_bn = nn.BatchNorm2d(int(d/2))
        self.deconv6 = nn.ConvTranspose2d(int(d/2), int(d/4), 5, 2, 0) # 41x41x64->85x85x32
        self.deconv6_bn = nn.BatchNorm2d(int(d/4))
        self.deconv7 = nn.ConvTranspose2d(int(d/4), int(d/8), 4, 2, 1) # 85x85x32->170x170x16
        self.deconv7_bn = nn.BatchNorm2d(int(d/8))
        self.deconv8 = nn.ConvTranspose2d(int(d/8), out_channel, 4, 2, 1) # 170x170x16->340x340x2

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        x = F.leaky_relu(self.deconv1_bn(self.deconv1(input)),0.2)
        x = F.leaky_relu(self.deconv2_bn(self.deconv2(x)),0.2)
        x = F.leaky_relu(self.deconv3_bn(self.deconv3(x)),0.2)
        x = F.leaky_relu(self.deconv4_bn(self.deconv4(x)),0.2)
        x = F.leaky_relu(self.deconv5_bn(self.deconv5(x)),0.2)
        x = F.leaky_relu(self.deconv6_bn(self.deconv6(x)),0.2)
        x = F.leaky_relu(self.deconv7_bn(self.deconv7(x)),0.2)
        x = F.leaky_relu(self.deconv8(x),0.2)
        return x

#%% Project the latent variables to the L2 unit ball for fast convergence
def project_l2_ball(z):
    z = z / np.maximum(np.sqrt(np.sum(z**2, axis=1))[:, np.newaxis], 1)
    return z

#%% Auxiliary 
def show_train_hist(hist, show = False, save = False, path = 'Train_hist.png'):
    x = range(len(hist['G_losses']))

    y1 = hist['G_losses']

    plt.plot(x, y1) #label='G_loss'

    plt.xlabel('Iter')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()

#%%
# training parameters
lr_g = 0.01
lr_z = 0.1
train_epoch = 50
siz_l = 60  #size of the latent variable

#%%
# initialize the latent variables z~N(0,1)
zi = np.random.randn(200, siz_l)
zi = project_l2_ball(zi)

# network
G = generator(siz_l)
G.weight_init(mean=0.0, std=0.02)
G.cuda()

z_ = torch.zeros((200, siz_l))
z_ = Variable(z_, requires_grad=True) #.cuda()


# Adam optimizer
optimizer = optim.Adam([
    {'params': G.parameters(), 'lr': lr_g},
    {'params': z_, 'lr': lr_z}
    ], betas=(0.7, 0.999))   #reduce the first beta value might be helpful

#Preparation for training
train_hist = {}
train_hist['G_losses'] = []
train_hist['per_epoch_ptimes'] = []
train_hist['total_ptime'] = []

#%% Start training
print('training start!')
start_time = time.time()
for epoch in range(train_epoch):
    G_losses = []
    epoch_start_time = time.time()
    
    zi = torch.FloatTensor(zi).view(-1,siz_l, 1, 1)
    zi = zi.cuda()
    z_.data = zi
    
    G_result = G(z_)
    
    #Plot part of the results for visualization
    test_image = G_result.cpu().data.numpy()
    test_image = test_image[:,0,:,:] + test_image[:,1,:,:]*1j
    
    fig = plt.figure(figsize=(20, 10))
    
    for k in range(5):
        plt.subplot(1, 5, k+1)
        plt.imshow(abs(test_image[k,:,:]), cmap='gray')
        plt.xticks([]);plt.yticks([])
    
    plt.tight_layout()
    plt.savefig('DMRI_result/results/MRI_' + str(epoch + 1) + '.png')
    plt.close()
    
    #To get the measurements from recon images
    G_s = G_result[:,None]
    AGs = A(G_s)
    
    G.zero_grad()
    
    #Using l2 loss
    loss = nn.MSELoss() #Add (reduction='sum') to become sum of square loss
    G_loss = loss(AGs,kdataT)
    
    G_loss.backward()
    optimizer.step()
    
    zi = project_l2_ball(z_.data.cpu().numpy())
        
    G_losses.append(G_loss.item())
    
    epoch_end_time = time.time()
    per_epoch_ptime = epoch_end_time - epoch_start_time
    
    print('[%d/%d] - ptime: %.2f, loss_g: %.3f' % ((epoch + 1), train_epoch, per_epoch_ptime, torch.mean(torch.FloatTensor(G_losses))))
    
    train_hist['G_losses'].append(torch.mean(torch.FloatTensor(G_losses)))
    train_hist['per_epoch_ptimes'].append(per_epoch_ptime)
    
end_time = time.time()
total_ptime = end_time - start_time
train_hist['total_ptime'].append(total_ptime)

print("Avg per epoch ptime: %.2f, total %d epochs ptime: %.2f" % (torch.mean(torch.FloatTensor(train_hist['per_epoch_ptimes'])), train_epoch, total_ptime))
print("Training finish!... save training results")
torch.save(G.state_dict(), "DMRI_result/generator_param.pkl")
with open('DMRI_result/train_hist.pkl', 'wb') as f:
    pickle.dump(train_hist, f)

show_train_hist(train_hist, save=True, path='DMRI_result/train_hist.png')