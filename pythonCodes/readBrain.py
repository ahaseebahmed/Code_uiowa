"""
Created on Tue Mar 24 12:15:06 2020

@author: haggarwal
"""

import os
import pathlib
import distro
import h5py as h5
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import miscTorch as sf
import viewShots as vs

def opA(org,csm,mask):
    tmp=org*csm
    tmp=sf.fft2c(tmp)
    b=tmp*mask
    return b
def opAt(b,csm,mask):
    tmp=mask*b
    tmp=sf.ifft2c(tmp)
    tmp=np.conj(csm)*tmp
    atb=np.sum(tmp,axis=-3)
    return atb

def pt_A(org,csm,mask):
    tmp=sf.pt_cpx_multipy(org,csm)
    tmp=sf.pt_fft2c(tmp)
    bT=mask*tmp
    return bT

def pt_At(b,csm,mask):
    tmp=mask*b
    tmp=sf.pt_ifft2c(tmp)
    csmConj=sf.pt_conj(csm)
    tmp=sf.pt_cpx_multipy(csmConj, tmp)
    atbT=torch.sum( tmp ,dim=-4)
    return atbT

class DataGen(Dataset):
    def __init__(self, readFrom,nFrom,nImg,acc,sigma,random=False):

        if distro.linux_distribution()[0]=='Ubuntu':
            fname='/var/tmp/hk/datasets/pi/dataset.hdf5'
        elif distro.linux_distribution()[0]=='CentOS Linux':
            fname='/Users/haggarwal/datasets/piData/dataset.hdf5'

        if random:
            self.sidx=np.sort(np.random.permutation(360)[0:nImg])
        else:
            self.sidx=np.r_[nFrom:nFrom+nImg]
        self.sigma=sigma
        self.readFrom=readFrom
        self.acc=acc
        self.nImg=nImg
        self.file = fname#h5.File(fname, 'r')


    def __len__(self):
        return self.nImg


    def __getitem__(self, idx):
        if self.readFrom=='trn':
            with h5.File(self.file,'r') as f:
                org= f['trnOrg'][self.sidx[idx]]
                csm= f['trnCsm'][self.sidx[idx]]
                mask=f['trnMask'][self.sidx[idx]]
        elif self.readFrom=='tst':
            with h5.File(self.file,'r') as f:
                org= f['tstOrg'][self.sidx[idx]]
                csm= f['tstCsm'][self.sidx[idx]]
                mask=f['tstMask'][self.sidx[idx]]
        org=sf.c2r(org)
        csm=sf.c2r(csm)
        mask=mask.astype(np.float32)

        org=torch.tensor(org)
        csm=torch.tensor(csm)
        mask=torch.tensor(mask)

        mask=sf.fftshift(mask,(-1,-2))
        mask=torch.repeat_interleave(mask[:,:,None],2,-1)

        A= lambda x: pt_A(x,csm,mask)
        At=lambda x: pt_At(x,csm,mask)
        b=A(org)
        noise=torch.randn(*b.shape,dtype=torch.float32)
        noise=noise*self.sigma
        b=b+noise
        atb=At(b)

        org=sf.pt_b2f(org)
        atb=sf.pt_b2f(atb)
        return org,atb#,csm,mask


class DataGenFast(Dataset):
    def __init__(self, readFrom,nFrom,nImg,acc,sigma,random=False):

        if distro.linux_distribution()[0]=='Ubuntu':
            fname='/var/tmp/hk/datasets/pi/dataset.hdf5'
        elif distro.linux_distribution()[0]=='CentOS Linux':
            fname='/Users/haggarwal/datasets/piData/dataset.hdf5'

        if random:
            sidx=np.sort(np.random.permutation(360)[0:nImg])
        else:
            sidx=np.r_[nFrom:nFrom+nImg]
        if readFrom=='trn':
            kyOrg,kyCsm,kyMask='trnOrg','trnCsm','trnMask'
        else:
            kyOrg,kyCsm,kyMask='tstOrg','tstCsm','tstMask'


        with h5.File(fname, 'r') as f:
            org=f[kyOrg][sidx]
            csm=f[kyCsm][sidx]
            mask=f[kyMask][sidx]

        self.nImg=org.shape[0]
        mask=np.fft.fftshift(mask,(-1,-2)).astype(np.float32)
        atb=np.zeros_like(org)
        for i in range(self.nImg):
            A=lambda x: opA(x, csm[i], mask[i])
            At=lambda x: opAt(x, csm[i], mask[i])

            b=A(org[i])
            noise=np.random.randn(*b.shape).astype(np.float32)
            noise=noise*sigma
            b=b+noise
            atb[i]=At(b)

        self.org=org
        self.atb=atb

    def __len__(self):
        return self.nImg


    def __getitem__(self, idx):
        org=self.org[idx]
        atb=self.atb[idx]
        org=torch.tensor(sf.c2r(org))
        atb=torch.tensor(sf.c2r(atb))
        org=sf.pt_b2f(org)
        atb=sf.pt_b2f(atb)
        return org,atb#,csm,mask




nFrom=50
nImg=1
acc=4
sigma=.01
datagen=DataGenFast('trn',nFrom,nImg,acc,sigma,True)
org,atb=datagen[0]
