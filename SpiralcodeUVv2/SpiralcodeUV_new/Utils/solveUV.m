function  x = solveUV(ktraj,w,kdata,csm, V, N, nIterations,SBasis,useGPU,factor)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

[nSamplesPerFrame,numFrames,~,~] = size(kdata);
[~,nbasis] = size(V);

if(useGPU)
    osf = 2; wg = 3; sw = 8;
    w=ones(nSamplesPerFrame*numFrames,1);
    %w=sqrt(abs(ktraj(:,:,1)));
    %w(w==0)=1e-4;
    %w=repmat(dcf,[1 5*numFrames]);
    ktraj_gpu = [real(ktraj(:)),imag(ktraj(:))]';
    FT = gpuNUFFT(ktraj_gpu,w(:),osf,wg,sw,[N,N],[],true);
    Atb = Atb_UV(FT,kdata,V,csm,N,true);
    if factor~=0
        Reg = @(x) reshape(reshape(x.*(0.5./(factor(:))),[N*N,nbasis])*SBasis,[N*N*nbasis,1]);
    else
        Reg = @(x) reshape(reshape(x,[N*N,nbasis])*SBasis,[N*N*nbasis,1]);
    end
    AtA = @(x) AtA_UV(FT,x,V,csm,N,nSamplesPerFrame) + Reg(x);
else
    FT= NUFFT(ktraj/N,1,0,0,[N,N]);
    Atb = Atb_UV(FT,kdata,V,csm,false);
    AtA = @(x) AtA_UV(FT,x,V,csm,nSamplesPerFrame);
end

x = pcg(AtA,Atb(:),1e-5,nIterations);
x = reshape(x,[N,N,nbasis]);

end

