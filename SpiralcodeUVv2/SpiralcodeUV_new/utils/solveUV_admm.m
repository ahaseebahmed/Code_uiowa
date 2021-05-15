function  U = solveUV_admm(ktraj,kdata,csm, V, N, Z,Y,dcf,useGPU)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

[nSamplesPerFrame,numFrames,~,~] = size(kdata);
[~,nbasis] = size(V);
x=ones(N,N,nbasis);
beta=0.1;

if(useGPU)
    
    osf = 2; wg = 3; sw = 8;
    %w=ones(nSamplesPerFrame*numFrames,1);
    w=repmat(dcf,[1 5*numFrames]);
    ktraj_gpu = [real(ktraj(:)),imag(ktraj(:))]';
    FT = gpuNUFFT(ktraj_gpu/N,w(:),osf,wg,sw,[N,N],[],true);
    Atb = Atb_UV(FT,kdata,V,csm,true) + beta*(Z-Y/beta);
    AtA = AtA_UV(FT,x,V,csm,nSamplesPerFrame)+beta*x;
    for ii=1:nbasis
        AtA1(:,:,ii)=inv(AtA(:,:,ii));
    end
    U   = Atb*reshape(AtA1,[N*N,nbasis]);

end


end

