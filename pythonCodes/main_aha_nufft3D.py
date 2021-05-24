"""
Created on Tue Apr 14 11:36:39 2020
This is the sense code in pytorch
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
import supportingFun as sf
import h5py as h5
from torchkbnufft.torchkbnufft import KbNufft
from torchkbnufft.torchkbnufft import AdjKbNufft
from torchkbnufft.torchkbnufft import MriSenseNufft
from torchkbnufft.torchkbnufft import AdjMriSenseNufft
from torchkbnufft.torchkbnufft.nufft.sparse_interp_mat import precomp_sparse_mats


gpu=torch.device('cuda:0')

directory='/Users/ahhmed/pytorch_sense/Konnor_3D/'
dtype = torch.float

NF=300 
nx=400
nch=2
nkpts=383
nintl=44
nbasis=4
lam=0.0
#%%

with h5.File(directory+'csm_300.h5', 'r') as f:  
  # coil sensitivity maps
  csm=np.asarray(f['/csm/re'])+np.asarray(f['/csm/im'])*1j
  csm=csm.astype(np.complex64)
  #csmTrn=np.transpose(csm,[0,2,1])
  #csm=np.fft.fftshift(np.fft.fftshift(csm,1),2)
  csmTrn=csm[0:nch,:,:]
  ncoils=csmTrn.shape[0]
  del csm
  
with h5.File(directory+'kdata.h5', 'r') as f:  
  # reading the RHS: AHb
  kdata=np.asarray(f['/kdata/re'])
  kdata=kdata+np.asarray(f['/kdata/im'])*1j
  kdata=kdata.astype(np.complex64)
  kdata=np.reshape(kdata,(7,NF,nintl*nkpts))
  kdata=np.squeeze(kdata[0:nch,0:NF,:])
  kdata=np.transpose(kdata,(1,0,2))
  #kdata=np.transpose(kdata,(1,0,2))  
#kdata=np.reshape(kdata,(nch,NF*6*2336)) 

#with h5.File(directory+'y.h5', 'r') as f:  
#  # reading the RHS: AHb
#  y=np.asarray(f['/y/re'])
#  y=y+np.asarray(f['/y/im'])*1j
#  y=y.astype(np.complex64)
#  y=np.squeeze(np.transpose(y[:,:,:],[0,1,2]))  
#kdata=np.reshape(kdata,(3,NF*6*2336)) 

with h5.File(directory+'dcf.h5', 'r') as f:  
  # reading the RHS: AHb
  dcf=np.asarray(f['/dcf/re'])
  dcf=np.reshape(dcf,(1,NF*nintl*nkpts))
  dcf=np.tile(dcf,(2,1))

with h5.File(directory+'ktraj.h5', 'r') as f: 
  ktraj=np.asarray(f['/ktraj/re'])
  #ktraj=ktraj+np.asarray(f['/csm/im'])*1j
  #ktraj=ktraj.astype(np.complex64)
  ktraj=np.squeeze(np.transpose(ktraj,[1,0]))*2*np.pi
ktraj=np.reshape(ktraj,(1,3,NF*nintl*nkpts))
#ktraj=np.transpose(ktraj,(0,1))  
#ktraj=np.tile(ktraj,(nbasis,1,1,))

with h5.File(directory+'L.h5', 'r') as f:  
  # reading the RHS: AHb
  L=np.asarray(f['/L'])
  U,sbasis,V=np.linalg.svd(L)
  V=V[NF-nbasis:NF,:]
  sbasis=np.diag(sbasis[NF-nbasis:NF,])*lam
Vn=np.zeros((nbasis,NF,NF))  
for i in range(nbasis):
    Vn[i,:,:]=np.diag(V[i,:])
#%%
im_size = csmTrn[0].shape

kdata = np.stack((np.real(kdata), np.imag(kdata)),axis=2)
# convert to tensor, unsqueeze batch and coil dimension
# output size: (1, 1, 2, ny, nx)
kdataT = torch.tensor(kdata).to(dtype)

dcfT = torch.tensor(dcf).to(dtype)
LT = torch.tensor(L).to(dtype)
VT = torch.tensor(V).to(dtype)
sbasis = torch.tensor(sbasis).to(dtype)

smap = np.stack((np.real(csmTrn), np.imag(csmTrn)), axis=1)
#smap=np.tile(smap,(NF,1,1,1,1,1))
smapT = torch.tensor(smap).to(dtype)
# convert to tensor, unsqueeze batch dimension
# output size: (1, 2, klength)
#ktraj = np.stack((np.real(ktraj), np.imag(ktraj)), axis=1)
ktrajT = torch.tensor(ktraj).to(dtype)

#y = np.stack((np.real(y), np.imag(y)), axis=0)
#yT = torch.tensor(y).to(dtype).unsqueeze(0).unsqueeze(0)

#%% take them to gpu
kdataT=kdataT.to(gpu)
smapT=smapT.to(gpu)
ktrajT=ktrajT.to(gpu)
dcfT=dcfT.unsqueeze(0).unsqueeze(0).to(gpu)
VT=VT.to(gpu)
sbasis=sbasis.to(gpu)
#%% generate atb
#sigma=0.0
#lam=1e+1
#cgIter=10
#cgTol=1e-15

nuf_ob = KbNufft(im_size=im_size).to(dtype)
nuf_ob=nuf_ob.to(gpu)
adjnuf_ob = AdjKbNufft(im_size=im_size).to(dtype)
adjnuf_ob=adjnuf_ob.to(gpu)
#real_mat, imag_mat = precomp_sparse_mats(ktrajT, adjnuf_ob)
#interp_mats = {'real_interp_mats': real_mat, 'imag_interp_mats': imag_mat}

#nufft_ob = MriSenseNufft(im_size=im_size, smap=smapT).to(dtype)
#nufft_ob=nufft_ob.to(gpu)
#adjnufft_ob = AdjMriSenseNufft(im_size=im_size, smap=smapT).to(dtype)
#adjnufft_ob=adjnufft_ob.to(gpu)
#nufft_ob = MriSenseNufft(smap=smapT,im_size=im_size).to(dtype)
#adjnufft_ob = AdjMriSenseNufft(smap=smapT,im_size=im_size ).to(dtype)

#At=lambda x: adjnufft_ob(x*dcfT,ktrajT,interp_mats)
#A=lambda x: nufft_ob(x,ktrajT,interp_mats)

#At=lambda x: adjnuf_ob(x,ktrajT)
#A=lambda x: nuf_ob(x,ktrajT)
#kdataT1=A(yT)
#%%
conj_smapT=smapT
conj_smapT[:,1]=conj_smapT[:,1]*-1
atbT=torch.zeros((nbasis,2,im_size[0],im_size[1],im_size[2])).to(gpu)
kdataT=torch.reshape(kdataT,(NF,nch,2*nintl*nkpts))
for i in range(nbasis):
    for j in range(nch):
        
        temp1=torch.diag(VT[i])@kdataT[:,j,:]
        temp1=torch.reshape(temp1,(NF,2,nintl*nkpts))
        temp1=temp1.permute(1,0,2)
        temp1=torch.reshape(temp1,(2,NF*nintl*nkpts)).unsqueeze(0).unsqueeze(0)
        atbT[i]=atbT[i]+(adjnuf_ob(temp1*dcfT,ktrajT)*conj_smapT[j]).squeeze(0)
    
del kdataT
del conj_smapT   
#%%
def AtA_UV(x):
    conj_smapT=smapT
    conj_smapT[:,1]=conj_smapT[:,1]*-1
    y=torch.zeros(nbasis,2,im_size[0],im_size[1],im_size[2]).to(gpu)
    for j in range(nch):
        virKsp=torch.zeros(NF,2*nintl*nkpts).to(gpu)
        for i in range(nbasis):
            temp=(smapT[j]*x[i]).unsqueeze(0).unsqueeze(0)
            ktb=nuf_ob(temp,ktrajT).squeeze(0).squeeze(0)
            ktb=torch.reshape(ktb,(2,NF,nintl*nkpts))
            ktb=ktb.permute(1,0,2)
            ktb=torch.reshape(ktb,(NF,2*nintl*nkpts))
            virKsp=virKsp+torch.diag(VT[i])@ktb
        for i in range(nbasis):
            temp=torch.diag(VT[i])@virKsp
            temp=torch.reshape(temp,(NF,2,nintl*nkpts))
            temp=temp.permute(1,0,2)
            temp=torch.reshape(temp,(2,NF*nintl*nkpts)).unsqueeze(0).unsqueeze(0)
            y[i]=y[i]+(adjnuf_ob(temp*dcfT,ktrajT)*conj_smapT[j]).squeeze(0)
    
    reg=sbasis@torch.reshape(x,(nbasis,2*im_size[0]*im_size[1]*im_size[2]))
    y=y+torch.reshape(reg,(nbasis,2,im_size[0],im_size[1],im_size[2]))
    return y

#%% pretrained deep learning network
    
def rhs(u1,D):
    y=[]
    nbct=int(NF/100)
    #u=u1.permute(3,1,2,0)
    #x=torch.matmul(u,D)
    #x=x.permute(3,0,1,2)
    u=torch.reshape(u1,(nbasis,nx*nx*2))
    x=torch.mm(D.T,u)
    x=torch.reshape(x,(NF,nx,nx,2))
    x=x.permute(0,3,1,2)
    xx=x.cpu().numpy()
    xx=np.fft.fftshift(np.fft.fftshift(xx,2),3)
    xx=np.reshape(xx,(nbct,100,2,nx,nx))
    xx=np.transpose(xx,(0,2,1,3,4))

    #trnDs=rd.DataGen(xx)
    trnDs=TensorDataset(torch.tensor(xx),torch.tensor(xx))
    #trnDs=rd.DataGen(xx,xx)
    ldr=DataLoader(trnDs,batch_size=1,shuffle=False,num_workers=8)
    with torch.no_grad():
        for data in (ldr):
            slcR=unet(data[0].to(gpu,dtype))
            y.append(slcR)
    Y=torch.cat(y)
    xx=Y.detach().cpu().numpy()
    xx=np.transpose(xx,(0,2,1,3,4))
    xx=np.reshape(xx,(NF,2,nx,nx))
    xx=np.fft.fftshift(np.fft.fftshift(xx,2),3)
    Y=torch.tensor(xx)
    Y=Y.to(gpu,dtype)
#    Y=Y.permute(2,3,1,0)
#    z=torch.matmul(Y,D.T)
#    z=z.permute(3,0,1,2)
    Y=Y.permute(0,2,3,1)
    Y=torch.reshape(Y,(NF,nx*nx*2))
    z=torch.mm(D,Y)
    z=torch.reshape(z,(nbasis,nx,nx,2))
    return z

def reg_term(u1,D):

    u=torch.reshape(u1,(nbasis,nx*nx*2))
    x=torch.mm(D.T,u)
    u2=torch.mm(D,x)
    u2=torch.reshape(u2,(nbasis,nx,nx,2))
    return u2 
#%% run cg-sense
lam2=1e-1
cgIter=3
cgTol=1e-15
recT=torch.zeros_like(atbT)

vt=VT
AtA=lambda x: AtA_UV(x)+lam2*reg_term(x,vt)      
#recT=torch.zeros_like(atbT)
atbT1=atbT
sf.tic()
for i in range(2):
    atbT1=atbT+lam2*rhs(recT,vt)
    recT=sf.pt_cg(AtA,atbT1,recT,cgIter,cgTol)
sf.toc()




#%%
rec = np.squeeze(recT.cpu().numpy())
rec = rec[:,0] + 1j*rec[:,1]

for i in range(100,180):
    plt.imshow((np.squeeze(np.abs(rec[3,:,:,i]))),cmap='gray')       
    plt.show() 
    plt.pause(0.01)
