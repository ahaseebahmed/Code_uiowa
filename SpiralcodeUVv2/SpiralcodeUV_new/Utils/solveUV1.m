function  x = solveUV1(ktraj,w,kdata,csm, V, N, nIterations,SBasis,useGPU,factor)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

[nSamplesPerFrame,numFrames,~,~] = size(kdata);
[~,nbasis] = size(V);
g_filt=fspecial('gaussian',[N,N],100);
ww=g_filt./max(g_filt(:));
ww=((1-ww))+1e-10;
ww=ww.^2;
%ww=fftshift(ww);
ww=repmat(ww,[1 1 nbasis-1]);
ww2=ww.*ww;


if(useGPU)
    osf = 2; wg = 3; sw = 8;
    %w=ones(nSamplesPerFrame*numFrames,1);
    %w=sqrt(abs(ktraj(:,:,1)));
    %w(w==0)=1e-4;
    %w=repmat(dcf,[1 5*numFrames]);
    ktraj_gpu = [real(ktraj(:)),imag(ktraj(:))]';
    FT = gpuNUFFT(ktraj_gpu,w(:),osf,wg,sw,[N,N],[],true);
    Atb = Atb_UV(FT,kdata.*repmat(sqrt(w),[1 1 size(csm,3)]),V,csm,N,true);
%     if factor~=0
%         Reg = @(x) reshape(reshape(x.*(0.5./(factor(:))),[N*N,nbasis])*SBasis,[N*N*nbasis,1]);
%     else
%         Reg = @(x) reshape((reshape(x,[N*N,nbasis]))*SBasis,[N*N*nbasis,1]);
%     end
    AtA = @(x) AtA_UV(FT,x,V,csm,N,nSamplesPerFrame) + Reg(x,ww2,SBasis,N,nbasis);
else
    FT= NUFFT(ktraj/N,1,0,0,[N,N]);
    Atb = Atb_UV(FT,kdata,V,csm,false);
    AtA = @(x) AtA_UV(FT,x,V,csm,nSamplesPerFrame);
end

x = pcg(AtA,Atb(:),1e-5,nIterations,[],[],Atb(:));
x = reshape(x,[N,N,nbasis]);

end

function xx=Reg(x,ww2,SBasis,N,nbasis)

tmp=reshape(x,[N*N,nbasis]);
tmp(:,1:end-1) = tmp(:,1:end-1).*reshape(ww2,[N^2 nbasis-1]);
tmp1=tmp*SBasis;
xx=reshape(tmp1,[N*N*nbasis,1]);

end

