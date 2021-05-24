#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 22:09:16 2020

@author: ahhmed
"""

import os,time
import numpy as np
import matplotlib.pyplot as plt
import torch

import supportingFun as sf

gpu=torch.device('cuda:0')
cpu=torch.device('cpu')

nx=512
nbasis=30
#%%
def ATBV(x,csmT,VT):
    
    nch=csmT.size(0)
    NF=VT.size(1)
    tmp1=torch.zeros((nx,nx,2)).to(gpu)
    atbv=torch.zeros((nx*nx*2,nbasis)).to(gpu)
    x=x.to(gpu)
    for i in range(NF):
        for j in range(nch):
            tmp=x[j,i]
            tmp=torch.reshape(tmp,(nx,nx,2))
            tmp1=tmp1+sf.pt_cpx_multipy(sf.pt_ifft2c(tmp),sf.pt_conj(csmT[j]))
            
        tmp2=VT[:,i].unsqueeze(1)
        tmp3=torch.reshape(tmp1,(nx*nx*2,1))
        atbv=atbv+tmp3@tmp2.T
        tmp1=torch.zeros((nx,nx,2)).to(gpu)
    
    atbv=atbv*nx
    atbv=atbv.permute(1,0)
    atbv=torch.reshape(atbv,(nbasis,nx,nx,2))
    #atbV=atbV.astype(np.complex64)
    #atbV=np.fft.fftshift(np.fft.fftshift(atbV,1),2)
    #atbV=np.transpose(atbV,[0,2,1])
    del tmp,tmp1,tmp2,tmp3
    return atbv
#%%
def ATBV_NFT(FT,kdata,csmT,VT):
    
    nch=csmT.size(0)
    N=csmT.size(1)
    nbas=VT.size(0)
    tmp1=torch.zeros((nx,nx,2)).to(gpu)
    atbv=torch.zeros((nx*nx*2,nbasis)).to(gpu)
    x=x.to(gpu)
    for i in range(nbas):
        for j in range(nch):
            tmp2=VT[:,i].unsqueeze(1)
            tmp=torch.diag(tmp2)@kdata[j,:,:]
            tmp=torch.reshape(tmp,(nx,nx,2))
            tmp1[i]=tmp1[i]+sf.pt_cpx_multipy(FT(tmp),sf.pt_conj(csmT[j]))
            
        
        tmp3=torch.reshape(tmp1,(nx*nx*2,1))
        atbv=atbv+tmp3@tmp2.T
        tmp1=torch.zeros((nx,nx,2)).to(gpu)
    
    atbv=atbv*nx
    atbv=atbv.permute(1,0)
    atbv=torch.reshape(atbv,(nbasis,nx,nx,2))
    #atbV=atbV.astype(np.complex64)
    #atbV=np.fft.fftshift(np.fft.fftshift(atbV,1),2)
    #atbV=np.transpose(atbV,[0,2,1])
    del tmp,tmp1,tmp2,tmp3
    return atbv
#%%

def maskV(msk,Vv):
    bsz=msk.size(0)
    res=torch.zeros((bsz,nbasis,nx*nx))
    res=res.to(gpu)
    for k in range(bsz):
        tmp3=msk[k].repeat(nbasis,1)#*tmp2
        res[k]=torch.diag(Vv[k])@tmp3
    return res


#%%
def maskV_cpu(msk,Vv):
    bsz=msk.size(0)
    res=torch.zeros((bsz,nbasis,nx*nx))
    res=res.to(gpu)
    msk=msk.to(gpu,dtype=torch.float32)
    for k in range(bsz):
        tmp3=msk[k].repeat(nbasis,1)#*tmp2
        res[k]=torch.diag(Vv[k])@tmp3
    return res.cpu()
#%%
def AtAUV(x,csmT,maskT1,VT):
#    atbv=torch.zeros(nbasis,nx,nx,2)
#    atbv=atbv.to(gpu)
#    tmp2=torch.zeros(nbasis,nch,nx,nx,2)
#    tmp2=tmp2.to(gpu)
    #tmp6=torch.zeros(nbasis,nx,nx,2)
    #tmp6=tmp6.to(gpu)
    nch=csmT.size(0)
    NF=maskT1.size(0)
    atbv=torch.cuda.FloatTensor(nbasis,nx,nx,2).fill_(0)
    tmp6=torch.cuda.FloatTensor(nbasis,nx,nx,2).fill_(0)
    csmConj=sf.pt_conj(csmT)

    for i in range(nch):
        #tmp=csmT[i,:,:,:]
        #tmp=tmp.repeat(nbasis,1,1,1)
        tmp1=sf.pt_cpx_multipy(x,csmT[i].repeat(nbasis,1,1,1))
        tmp2=sf.pt_fft2c(tmp1)
        del tmp1
        for k in range(NF):
            #tmp=maskT[k,:,:,:]
            #tmp=tmp.repeat(nbasis,1,1,1).to(gpu)
            tmp3=maskT1[k].unsqueeze(2).repeat(nbasis,1,1,2)*tmp2
            #tmp3=tmp3.to(gpu,dtype)
            tmp4=VT[:,k].unsqueeze(1)
            tmp3=torch.reshape(tmp3,(nbasis,nx*nx*2))
            tmp5=tmp4.T@tmp3#torch.mm(tmp4.T,tmp3)
            #tmp5=torch.matmul(tmp3.permute(1,2,3,0),tmp4)
            #tmp5=torch.matmul(tmp5,tmp4.T)
            tmp5=tmp4@tmp5#torch.mm(tmp4,tmp5)        
            tmp5=torch.reshape(tmp5,(nbasis,nx,nx,2))
            #tmp5=tmp5.permute(3,0,1,2)
            tmp6=tmp6+tmp5
        del tmp2,tmp3,tmp4,tmp5   
        tmp1=sf.pt_ifft2c(tmp6)
        #tmp=csmConj[i,:,:,:]
        #tmp=tmp.repeat(nbasis,1,1,1).to(gpu)
        tmp2=sf.pt_cpx_multipy(csmConj[i].repeat(nbasis,1,1,1),tmp1)
        atbv=atbv+tmp2
        #tmp6=torch.zeros(nbasis,nx,nx,2)
        #tmp6=tmp6.to(gpu)
        tmp6=tmp6.fill_(0)
        del tmp1,tmp2

    x=torch.reshape(x,(nbasis,nx*nx*2))
    #x=W*x
    #reg=torch.mm(sT,x)
    #reg=torch.reshape(reg,(nbasis,nx,nx,2))
    #atbv=atbv+reg
    del x
    return atbv