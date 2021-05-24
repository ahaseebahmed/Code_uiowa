#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 15:51:05 2021

@author: ahhmed
"""

import numpy as np
import scipy.io
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataset import Dataset
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from torch.utils.dlpack import to_dlpack
from torch.utils.dlpack import from_dlpack
#from numba import jit, njit, prange, cuda
#from joblib import Parallel, delayed
import os

def run_Bloch_equations(T1s, TRs, flip_angles):
#     print('Min T1 used in Bloch equations:', torch.min(T1s))
#     print('Max T1 used in Bloch equations:', torch.max(T1s))
#     print('Min T2 used in Bloch equations:', torch.min(T2s))
#     print('Max T2 used in Bloch equations:', torch.max(T2s))
    
    device = torch.device('cuda:0')

   

    num_elements = T1s.shape[0]
    T1s = torch.tensor(T1s).to(device)
    sequence_length = flip_angles.shape[0]
    dictionary = torch.zeros([sequence_length,num_elements])
    
    state_matrix = torch.zeros([3,num_elements],dtype=torch.float64, device=device)
    state_matrix[2,:] = 1    # Equilibrium magnetization
    
    state_matrix = state_matrix.to(device)

    #R1s = 1/T1s
    #E1 = torch.exp(-TRs/T1s.T)
    #E1 = E1.to(device)
    
        
    for l in range(sequence_length):
        excitation = np.array([[(np.cos(flip_angles[l])), 0, np.sin(flip_angles[l])],
                        [0, 1, 0],
                        [-np.sin(flip_angles[l]), 0, np.cos(flip_angles[l])]])
        excitation = torch.tensor(excitation,dtype=torch.float64, device=device)
        state_matrix = excitation@state_matrix
        state_matrix[2,:] = 1 - (1-state_matrix[2,:])*torch.exp(-TRs[l]/T1s)
        dictionary[l,:] = state_matrix[0,:]   # Signal is F0 state.
        state_matrix[0:1,:] = 0

    return dictionary