import numpy as np
import torch
import time

def tor_conj(x):
    y = x.clone()
    y[:,:,1,:,:] = x[:,:,1,:,:]*(-1)
    return y

def tor_dot_product(x,y):
    return torch.stack((x[:,:,0,:,:]*y[:,:,0,:,:]-x[:,:,1,:,:]*y[:,:,1,:,:], x[:,:,1,:,:]*y[:,:,0,:,:]+x[:,:,0,:,:]*y[:,:,1,:,:]),dim=-1)

def tor_conjgrad(Aop,rhs,ini,max_iter,max_eps):
    """
    Conjugate Gradient Algorithm applied to batches; assumes the first index is batch size.
    
    Args:
    ini (Tensor): The initial input to the algorithm.
    rhs (Tensor): The right hand side of the system of linear equations
    Aop (func): A function performing the normal equations
    max_iter (int): Maximum number of times to run conjugate gradient descent
    max_eps (float): Determines how small the residuals must be before termination
    
    Returns:
    	A tuple containing the output Tensor x
    """
    fn = lambda a,b: torch.sum(tor_dot_product(tor_conj(a),b))
    x = ini
    r = rhs - Aop(x)
    i,p = 0,r
    rTr=fn(r,r)
    eps=torch.tensor(1e-10)
    
    for i in range(max_iter):
        alpha = rTr/(fn(p,Aop(p))+eps)
        x = x + alpha * p
        r = r - alpha * Aop(p)
        rTrNew = fn(r,r)
        if torch.sqrt(rTrNew+eps) < max_eps:
            break
        beta = rTrNew / (rTr+eps)
        p = r + beta * p
        rTr=rTrNew.clone()
    
    return x


#%%
def TicTocGenerator():
    ti = 0           # initial time
    tf = time.time() # final time
    while True:
        ti = tf
        tf = time.time()
        yield tf-ti # returns the time difference

TicToc = TicTocGenerator() # create an instance of the TicTocGen generator

# This will be the main function through which we define both tic() and toc()
def toc(tempBool=True):
    # Prints the time difference yielded by generator instance TicToc
    tempTimeInterval = next(TicToc)
    if tempBool:
        print( "Elapsed time: %f seconds.\n" %tempTimeInterval )

def tic():
    # Records a time in TicToc, marks the beginning of a time interval
    toc(False)
