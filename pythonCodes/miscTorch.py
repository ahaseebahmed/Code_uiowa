"""
Created on Sat Mar  7 19:53:08 2020
This file is adapted from fastMRI transforms.py file.
@author: haggarwal
"""

#import numpy as np
import time
import torch
import mkl_fft
import numpy as np
#from skimage.metrics import structural_similarity


#%%
def c2r(inp):
    return np.stack( (inp.real,inp.imag),axis=-1)

def r2c(inp):
    return inp[...,0] + 1j*inp[...,1]

def pt_b2f(inp):
    if inp.ndim==3:
        return inp.permute(2,0,1)
    elif inp.ndim==4:
        return inp.permute(0,3,1,2)
    elif inp.ndim==5:
        return inp.permute(0,4,1,2,3)
    else:
        print('wrong dimensions')

def pt_f2b(inp):
    if inp.ndim==3:
        return inp.permute(1,2,0)
    elif inp.ndim==4:
        return inp.permute(0,2,3,1)
    elif inp.ndim==5:
        return inp.permute(0,2,3,4,1)
    else:
        print('wrong dimensions')



#%%
def pt_abs(data):
    #if data.shape[-1] != 2:
    #    print('wrong shape')
    val=(data**2).sum(dim=-1).sqrt()
    return val

def pt_conj(inp):
    out=inp.clone()
    out[...,1]=inp[...,1]*-1
    return out

def pt_cpx_multipy(w,z):
    #w=u+jv
    #z=x+jy
    #w*z= (ux-vy) + i(vx+uy)=a+ib
    a=w[...,0]*z[...,0]-w[...,1]*z[...,1]
    b=w[...,1]*z[...,0]+w[...,0]*z[...,1]
    c=torch.stack((a,b),dim=-1)
    return c

def pt_cpx_matmul(A,B):
    '''A: mxnx2, B: nxkx2, C: mxkx2. It supports broadcasting'''
    nrows=A.shape[-3]
    ncols=B.shape[-2]
    def makeRealMatrix(ww):
        w1=torch.cat((ww[...,0],-1*ww[...,1]),dim=-1)
        w2=torch.cat((ww[...,1],   ww[...,0]),dim=-1)
        wr=torch.cat((w1,w2),dim=-2)
        return wr
    Ar=makeRealMatrix(A)
    Br=makeRealMatrix(B)
    Cr=Ar@Br

    Creal=Cr[...,0:nrows,0:ncols]
    Cimag=Cr[...,nrows:,0:ncols]
    C=torch.stack([Creal,Cimag],dim=-1)
    return C
#%%
def pt_fft2c(data):
    assert data.size(-1) == 2
    data = ifftshift(data, dim=(-3, -2))
    data = torch.fft(data, 2, normalized=True)
    data = fftshift(data, dim=(-3, -2))
    return data

def pt_ifft2c(data):
    assert data.size(-1) == 2
    data = ifftshift(data, dim=(-3, -2))
    data = torch.ifft(data, 2, normalized=True)
    data = fftshift(data, dim=(-3, -2))
    return data

def fftshift(x, dim=None):
    """
    Similar to np.fft.fftshift but applies to PyTorch Tensors
    """
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [dim // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = x.shape[dim] // 2
    else:
        shift = [x.shape[i] // 2 for i in dim]
    return roll(x, shift, dim)


def ifftshift(x, dim=None):
    """
    Similar to np.fft.ifftshift but applies to PyTorch Tensors
    """
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [(dim + 1) // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = (x.shape[dim] + 1) // 2
    else:
        shift = [(x.shape[i] + 1) // 2 for i in dim]
    return roll(x, shift, dim)


def roll(x, shift, dim):
    """
    Similar to np.roll but applies to PyTorch Tensors
    """
    if isinstance(shift, (tuple, list)):
        assert len(shift) == len(dim)
        for s, d in zip(shift, dim):
            x = roll(x, s, d)
        return x
    shift = shift % x.size(dim)
    if shift == 0:
        return x
    left = x.narrow(dim, 0, x.size(dim) - shift)
    right = x.narrow(dim, x.size(dim) - shift, shift)
    return torch.cat((right, left), dim=dim)

#%%
def fft2c(img):
    """
    it works on last two dimensions. takes image domain data and do the
    fft2 to return kspace data
    """
    shp=img.shape
    nimg=int(np.prod(shp[0:-2]))
    scale=1/np.sqrt(np.prod(shp[-2:]))
    img=np.reshape(img,(nimg,shp[-2],shp[-1]))

    tmp=np.empty_like(img,dtype=np.complex64)
    for i in range(nimg):
        #tmp[i]=scale*np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img[i])))
        tmp[i]=scale*np.fft.fftshift(mkl_fft.fft2(np.fft.ifftshift(img[i])))

    kspace=np.reshape(tmp,shp)
    return kspace


def ifft2c(kspace):
    """
    it works on last two dimensions. takes image domain data and do the
    fft2 to return kspace data
    """
    shp=kspace.shape
    scale=np.sqrt(np.prod(shp[-2:]))
    nimg=int(np.prod(shp[0:-2]))

    kspace=np.reshape(kspace,(nimg,shp[-2],shp[-1]))

    tmp=np.empty_like(kspace)
    for i in range(nimg):
        #tmp[i]=scale*np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(kspace[i])))
        tmp[i]=scale*np.fft.fftshift(mkl_fft.ifft2(np.fft.ifftshift(kspace[i])))

    img=np.reshape(tmp,shp)
    return img
#%%
def maxmin(x):
    if np.iscomplexobj(x):
        y= np.asarray([ [np.max(np.abs(x)), np.min(np.abs(x))],
             [np.max(np.real(x)),np.min(np.real(x))],
             [np.max(np.imag(x)),np.min(np.imag(x))] ])
        return y
    else:
        return  x.max(),x.min()
#%%
def subMeanStd(img,mn,st):
    return (img-mn)/(st+1e-10)

def pt_normalizeMeanStd(img):
    #img: batchxchnxHxW
    tmp=pt_abs(pt_f2b(img))
    mn=torch.mean(tmp,dim=(-1,-2),keepdim=True).unsqueeze(-3)
    st=torch.std (tmp,dim=(-1,-2),keepdim=True).unsqueeze(-3)
    mn=torch.repeat_interleave(mn,2,dim=-3)
    st=torch.repeat_interleave(st,2,dim=-3)
    return subMeanStd(img,mn,st),mn,st

def pt_normalizeMeanStd3d(img):
    #img: batchxchnxHxW
    tmp=pt_abs(pt_f2b(img))
    mn=torch.mean(tmp,dim=(-1,-2,-3),keepdim=True).unsqueeze(-4)
    st=torch.std (tmp,dim=(-1,-2,-3),keepdim=True).unsqueeze(-4)
    mn=torch.repeat_interleave(mn,2,dim=-4)
    st=torch.repeat_interleave(st,2,dim=-4)
    return subMeanStd(img,mn,st),mn,st

def pt_normalize01(img,eps=1e-10):
    mx=torch.max(img)
    mn=torch.min(img)
    return (img-mn)/(mx-mn+eps)

def pt_divide_by_max_abs(img,eps=1e-10):
    return img/torch.max(pt_abs(img)+eps)

def normalize01(img):
    """
    Normalize the image between 0 and 1
    """
    shp=img.shape
    if np.ndim(img)>=3:
        nimg=np.prod(shp[0:-2])
    elif np.ndim(img)==2:
        nimg=1
    img=np.reshape(img,(nimg,shp[-2],shp[-1]))
    eps=1e-15
    img2=np.empty_like(img)
    for i in range(nimg):
        mx=img[i].max()
        mn=img[i].min()
        img2[i]= (img[i]-mn)/(mx-mn+eps)

    img2=np.reshape(img2,shp)
    return img2
#%%
def myPSNR(org,recon):
    sqrError=np.abs(org-recon)**2
    N=np.prod(org.shape[-2:])
    mse=np.sum(sqrError,axis=(-1,-2))/N
    maxval=np.max(org,axis=(-1,-2)) + 1e-15
    psnr=10*np.log10(maxval**2/(mse+1e-15 ))

    return psnr

#def mySSIM(org,rec):
#    """
#    org and rec are 3D arrays in range [0,1]
#    """
#    shp=org.shape
#    if np.ndim(org)>=3:
#        nimg=np.prod(shp[0:-2])
#    elif np.ndim(org)==2:
#        nimg=1
#    org=np.reshape(org,(nimg,shp[-2],shp[-1]))
#    rec=np.reshape(rec,(nimg,shp[-2],shp[-1]))

#    ssim=np.empty((nimg,),dtype=np.float32)
#    for i in range(nimg):
#        ssim[i]=structural_similarity(org[i],rec[i],data_range=org.max())
#    return ssim


#%%
def getCoilCombined(img,csm):
    nSlice= csm.shape[0]
    eps=1e-8;
    csmWt=np.empty_like(csm)
    for s in range(nSlice):
        tmp=csm[s]
        csmsos=np.sum(np.abs(tmp)**2,axis=0)
        tmpmask=csmsos>eps;
        x=1/(csmsos+np.logical_not(tmpmask))
        csmWt[s]=tmpmask*(np.conj(tmp)*x)

    img=np.sum(img*csmWt,axis=1)
    return img
#%%
def pt_cpx_crop(data, shape):
    assert 0 < shape[0] <= data.shape[-3]
    assert 0 < shape[1] <= data.shape[-2]
    w_from = (data.shape[-3] - shape[0]) // 2
    h_from = (data.shape[-2] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]
    return data[..., w_from:w_to, h_from:h_to, :]

def crop(data, shape=(320,320)):

    w_from = (data.shape[-2] - shape[0]) // 2
    h_from = (data.shape[-1] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]
    return data[..., w_from:w_to, h_from:h_to]
#%%
def sos(data,dim=-3):
    res= np.sqrt(np.sum(np.abs(data)**2,dim))
    return res
#%%
def TicTocGenerator():
    # Generator that returns time differences
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
