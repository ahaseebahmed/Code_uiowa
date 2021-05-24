#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 15:36:39 2021

@author: ahhmed
"""

#import cupy as cp
import numpy as np
import scipy.io
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from torch.utils.data.dataset import Dataset
import matplotlib.pyplot as plt
#from sklearn.preprocessing import normalize
#from torch.utils.dlpack import to_dlpack
#from torch.utils.dlpack import from_dlpack
#from numba import jit, njit, prange, cuda
#from joblib import Parallel, delayed
#import os
#import sys
from helperfunctions_aha import run_Bloch_equations


def giveFingerPrintSVD(params,nsvd):


    device = params["device"]
    nSpirals = params["nSpirals"]
    IRreprate = params["IRreprate"]
    flip = params["flip"]
    nIR = int(nSpirals/IRreprate)

    flip_angles = 3.142/180*flip*np.ones((nSpirals+nIR))
    TRs = params["TR"]/1000*np.ones((nSpirals+nIR))

    for n in np.arange(0,nSpirals-1,IRreprate+1):
        flip_angles[n] = 3.142    
        TRs[n] = 1e-10

    TRs = torch.tensor(TRs).to(device)


    T1s_sec = np.arange(0,5,0.01)
    fingerprints = run_Bloch_equations(T1s_sec.squeeze(), TRs, flip_angles)

    fpred = torch.zeros(IRreprate*nIR,fingerprints.shape[1])
    n=0
    m=1
    for i in range(nIR):
        fpred[n:n+IRreprate,:] = fingerprints[m:m+IRreprate,:]
        n=n+IRreprate
        m=m+IRreprate + 1

    nSpirals = params["nSpirals"]
    nintl = params['nintlPerFrame']
    nsamp = nSpirals*nintl

    #fpred = fpred[800*8-nsamp:800*8,:]

    fpred = np.reshape(fpred,(params['nFramesDesired'],nintl,fingerprints.shape[1]))
    fpred = torch.mean(fpred,1)
    fig,p = plt.subplots()

    fpred = fpred.cpu().numpy()
    p.plot(fpred)
    plt.grid()
    u,s,v = np.linalg.svd(fpred)
    ur = u[:,0:nsvd]

    return (ur,fpred)