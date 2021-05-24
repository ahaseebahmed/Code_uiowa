"""
Created on Tue Apr 14 11:43:44 2020

@author: haggarwal
"""
import numpy as np
import torch
import time
#from skimage.metrics import structural_similarity
#%%

nbasis=30
nx=512
nf=900
NF=200


#%%
def r2c(inp):
    return inp[...,0] + 1j*inp[...,1]

def c2r(inp):
    return np.stack( (inp.real,inp.imag),axis=-1)

def pt_abs(data):
    val=(data**2).sum(dim=-1).sqrt()
    return val

def pt_conj(inp):
    out=inp.clone()
    out[:,:,1]=inp[:,:,1]*-1
    return out

#def pt_cpx_multipy(w,z):
#    return torch.stack((w[...,0]*z[...,0]-w[...,1]*z[...,1],w[...,1]*z[...,0]+w[...,0]*z[...,1]),dim=-1)

def pt_cpx_multipy(w,z):
    return torch.stack((w[:,:,0]*z[:,:,0]-w[:,:,1]*z[:,:,1],w[:,:,1]*z[:,:,0]+w[:,:,0]*z[:,:,1]),dim=2)

#def pt_cpx_multipy(t1,t2):
#    return torch.view_as_complex(torch.stack((t1.real @ t2.real - t1.imag @ t2.imag, t1.real @ t2.imag + t1.imag @ t2.real),dim=2))
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

def complex_mult(a, b, dim=0):
    """Complex multiplication, real/imag are in dimension dim.

    Args:
        a (tensor): A tensor where dimension dim is the complex dimension.
        b (tensor): A tensor where dimension dim is the complex dimension.
        dim (int): An integer indicating the complex dimension.

    Returns:
        tensor: a * b, where * executes complex multiplication.
    """
    assert a.shape[dim] == 2
    assert b.shape[dim] == 2

    real_a = a.select(dim, 0)
    imag_a = a.select(dim, 1)
    real_b = b.select(dim, 0)
    imag_b = b.select(dim, 1)

    c = torch.stack(
        (real_a*real_b - imag_a*imag_b, imag_a*real_b + real_a*imag_b),
        dim
    )

    return c
#%%
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
#%%
def pt_cg(A,rhs,ini,cgIter,cgTol):
    fn=lambda a,b: torch.sum(pt_cpx_multipy(pt_conj(a),b))
    x=ini#torch.zeros_like(rhs)
    r=rhs-A(x)
    i,p=0,r
    rTr=fn(r,r)
    eps=torch.tensor(1e-10)
    for i in range(cgIter):
        #Ap=A(p)
        alpha=rTr/(fn(p,A(p))+eps)
        x = x + alpha * p
        r = r - alpha * A(p)
        rTrNew = fn(r,r)
        if torch.sqrt(rTrNew+eps) < cgTol:
            break
        beta = rTrNew / (rTr+eps)
        p = r + beta * p
        rTr=rTrNew.clone()
    return x#,torch.sqrt(rTrNew+eps)
#%%
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
    psnr=10*np.log10(maxval**2/(mse+1e-15))

    return psnr
#%%
def myPSNR1(org,recon):
    snr=np.zeros((200,1))
    for i in range(200):
        t1=np.abs(org[i]-recon[i])
        mse=np.linalg.norm(t1,'fro')
        mse=mse**2
        t2=np.linalg.norm(org[i],'fro')
        t2=t2**2
        snr[i]=10*np.log10(t2/mse)
    return snr

#%%
def scale(I,u):
    for i in range(200):
        t1=u[i]
        t2=I[i]
        alpha=np.sum(t1*t2)/np.sum(t1*t1)
        u[i]=alpha*t1
    return u
        

#%%
def myPSNR_dynamic(org,v,recon,v1):
    org=org.view(nbasis,-1)
    recon=recon.view(nbasis,-1)
    orgD=v.T@org
    reconD=v1.T@recon
    orgD=orgD.view(NF,nx,nx,2)
    reconD=reconD.view(NF,nx,nx,2)

    #orgDM=torch.sqrt(orgD[:,:,:,0]**2+orgD[:,:,:,1]**2)
    orgDM=torch.sqrt(orgD[:,100:400,200:400,0]**2+orgD[:,100:400,200:400,1]**2)
    #reconDM=torch.sqrt(reconD[:,:,:,0]**2+reconD[:,:,:,1]**2)
    reconDM=torch.sqrt(reconD[:,100:400,200:400,0]**2+reconD[:,100:400,200:400,1]**2)
    orgDM=orgDM/torch.max(orgDM)
    reconDM=reconDM/torch.max(reconDM)
    sqrError=(orgDM-reconDM)**2
    N=300*200
    mse=torch.sum(sqrError,axis=(-1,-2))
    mse=mse/N
    tmp=orgDM.view(NF,-1).cpu().numpy()
    maxval=np.max(tmp,axis=1)**2+1e-15
    maxval=torch.tensor(maxval).cuda()
    psnr=10*torch.log10(maxval/(mse+1e-15 ))
    psnr_m=torch.mean(psnr)
    return psnr_m.item()

#%%
def myPSNR_dynamic1(org,v,recon,v1):
    #N=nx*nx
    snr=0
    snr=torch.tensor(snr)
    org=org.view(nbasis,-1)
    recon=recon.view(nbasis,-1)
    orgD=v.T@org
    reconD=v1.T@recon
    orgD=orgD.view(NF,nx,nx,2)
    reconD=reconD.view(NF,nx,nx,2)

    orgDM=torch.sqrt(orgD[:,100:400,200:400,0]**2+orgD[:,100:400,200:400,1]**2)
    reconDM=torch.sqrt(reconD[:,100:400,200:400,0]**2+reconD[:,100:400,200:400,1]**2)
    orgDM=orgDM/torch.max(orgDM)
    reconDM=reconDM/torch.max(reconDM)
    sqrError=abs(orgDM-reconDM)
    for i in range(NF):
        sqnorm=1e-15+torch.norm(sqrError[i],p="fro")**2
        ornorm=torch.norm(orgDM[i],p="fro")**2
        snr=snr+10*torch.log10(ornorm/sqnorm)

    psnr_m=snr/NF
    return psnr_m.item()
#%%
def complex_multiplication(x,L,NF):
    tmp=x.cpu().numpy().squeeze(1)
    tmp1=tmp[:,0,]+1j*tmp[:,1,]
    tmp1=np.reshape(tmp1,(NF,340*340))
    tmp1=np.transpose(tmp1)
    yy=np.matmul(tmp1,L)
    yy=np.transpose(yy)
    yy=np.reshape(yy,(NF,340,340))
    yy = np.stack((np.real(yy), np.imag(yy)), axis=1)
    yyT=torch.tensor(yy).unsqueeze(1)
    
    #re=atbT[:,:,0,].squeeze(1).view(200,-1)
    #im=x[:,:,1,].squeeze(1).view(200,-1)
    #re_f=torch.matmul(L,re)
    #im_f=torch.matmul(L,im)
    #tmp=torch.reshape(x,(NF,1*2*340*340))
    #tmp=torch.matmul(L,tmp)
    #res=torch.reshape(tmp,(NF,1,2,340,340))
#    res=torch.zeros((200,2,340*340))    
#    res[:,0]=re_f
#    res[:,1]=im_f
#    res=res.unsqueeze(1)
#    res=res.view(200,1,2,340,340)
    
    return yyT

#%%
def basis_multi(x,sb):
    tmp=x.permute(1,2,3,4,0)
    tmp=torch.reshape(tmp,(1*2*340*340,30))
    tmp1=tmp@sb
    tmp1=torch.reshape(tmp1,(1,2,340,340,30))
    y=tmp1.permute(4,0,1,2,3)
    
    return y
#%%
#def AtA_UV(FT,FTt,x,ktrajT,V,im_size,nB,NF,nch,nintl,nkpts,dcf):
#    
#    ktb=FT(x,ktrajT)
#    ktb=torch.reshape(ktb,(nB,nch,2,NF,nintl*nkpts))
#    ktb=ktb.permute(3,0,1,2,4)
#    ktb=torch.reshape(ktb,(NF,nB,nch*2*nintl*nkpts))
#    
#    tmp1=torch.zeros((NF,nch*2*nintl*nkpts)).to('cuda:0')
#    
#    for i in range(nB):
#        tmp1=tmp1+V[i,]@ktb[:,i,:]
#        #tmp1=tmp1+torch.matmul(V[i,],ktb[:,i,:])
#        
#    V=torch.reshape(V,(nB*NF,NF))
#    tmp2=V@tmp1
#    tmp2=torch.reshape(tmp2,(nB,NF,nch,2,nintl*nkpts))
#    tmp2=tmp2.permute(0,2,3,1,4)
#    tmp2=torch.reshape(tmp2,(nB,nch,2,NF*nintl*nkpts))
#    atb1=FTt(tmp2*dcf,ktrajT)
#    
#    return atb1
#%%
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
#
#    ssim=np.empty((nimg,),dtype=np.float32)
#    for i in range(nimg):
#        ssim[i]=structural_similarity(org[i],rec[i],data_range=org.max())
#    return ssim
#%%
def AtAUV_fast(x,csmT,csmConj,maskT,W,sT):
    nbasis=len(x)
    nx=len(x[0])
    nch=len(x)
    atbv=torch.cuda.FloatTensor(nbasis,nx,nx,2).fill_(0)
    for i in range(nch):
        tmp2=pt_fft2c(pt_cpx_multipy(x,csmT[i].repeat(nbasis,1,1,1)))
        tmp2=torch.reshape(tmp2,(nbasis,nx*nx*2))
        tmp2=tmp2.repeat(nbasis,1,1)*maskT
        tmp=tmp2.sum(axis=1)
        tmp=torch.reshape(tmp,(nbasis,nx,nx,2))
        tmp2=pt_cpx_multipy(csmConj[i].repeat(nbasis,1,1,1),pt_ifft2c(tmp))
        atbv=atbv+tmp2
        del tmp2
    x=torch.reshape(x,(nbasis,nx*nx*2))
    x=W*x
    reg=torch.mm(sT,x)
    reg=torch.reshape(reg,(nbasis,nx,nx,2))
    atbv=atbv+reg
    return atbv
#%%
    
def subMeanStd(img,mn,st):
    return (img-mn)/(st+1e-10)
def pt_abs(data):
    #if data.shape[-1] != 2:
    #    print('wrong shape')
    val=(data**2).sum(dim=-1).sqrt()
    return val
   
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
        
def subMeanStd(img,mn,st):
    return (img-mn)/(st+1e-10)

def pt_normalizeMeanStd(img):
    #img: batchxchnxHxW
    tmp=pt_abs(pt_f2b(img))
    mn=torch.mean(tmp,dim=(-1,-2),keepdim=True).unsqueeze(-3)
    st=torch.std (tmp,dim=(-1,-2),keepdim=True).unsqueeze(-3)
    mn=torch.repeat_interleave(mn,60,dim=-3)
    st=torch.repeat_interleave(st,60,dim=-3)
    return subMeanStd(img,mn,st),mn,st
#%%
def reg_term(u1,D):
    nbasis=len(u1)
    nx=len(u1[0])
    u=torch.reshape(u1,(nbasis,nx*nx*2))
    x=torch.mm(D.T,u)
    u2=torch.mm(D,x)
    u2=torch.reshape(u2,(nbasis,nx,nx,2))
    return u2 
#%%
def D_C():
    
    atbT1=atbT+lam2*rhs(recT,vt)
    recT=pt_cg(AtA,atbT1,recT,4,cgTol)
#%%
def bas2img(self,u1,D):
    nbasis=len(u1)
    nx=len(u1[0])
    u1=torch.reshape(u1,(nbasis,nx*nx*2))
    x=torch.mm(D.T,u1)
    x=torch.reshape(x,(10,nx,nx,2))
    x=x.permute(0,3,1,2)
    return x

def img2bas(self,x,D):        
    x=x.permute(0,2,3,1)
    x=torch.reshape(x,(10,nx*nx*2))
    z=torch.mm(D,x)
    z=torch.reshape(z,(nbasis,nx,nx,2))
    return z