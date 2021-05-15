function  x = solveUV1(ktraj,kdata,csm, V, N, nIterations,SBasis,useGPU)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

[nSamplesPerFrame,numFrames,~,~] = size(kdata);
[~,nbasis] = size(V);

if(useGPU)
    osf = 2; wg = 3; sw = 8;
    w=ones(nSamplesPerFrame*numFrames,1);
    %w=repmat(dcf,[1 5*numFrames]);
    ktraj_gpu = [real(ktraj(:)),imag(ktraj(:))]';
    FT = gpuNUFFT(ktraj_gpu,w(:),osf,wg,sw,[N,N],[],true);
    Atb = Atb_UV(FT,kdata,V,csm,true);
    Reg = @(x) reshape(reshape(x,[N*N,nbasis])*SBasis,[N*N*nbasis,1]);
    AtA = @(x) AtA_UV(FT,x,V,csm,nSamplesPerFrame) + Reg(x);
else
    %kdata=reshape(kdata,2496,4,numFrames,6);
    ktraj=reshape(ktraj,2496,4,numFrames);
    FT= NUFFT(ktraj/N,1,0,0,[N,N]);
    Atb = Atb_UV(FT,kdata,V,csm,false);
    Reg = @(x) reshape(reshape(x,[N*N,nbasis])*SBasis,[N*N*nbasis,1]);
    AtA = @(x) AtA_UV(FT,x,V,csm,nSamplesPerFrame)+Reg(x);
end

x = pcg(AtA,Atb(:),1e-5,nIterations);
x = reshape(x,[N,N,nbasis]);

end

