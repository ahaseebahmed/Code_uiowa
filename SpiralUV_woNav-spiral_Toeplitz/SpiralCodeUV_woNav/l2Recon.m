function data = l2Recon(kdata,ktraj,coilImages,lambda)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here

N = ceil(max([max(abs(imag(ktraj(:)))),max(abs(imag(ktraj(:))))]));
csm = giveEspiritMapsSmall(coilImages,N,N);
[~,~,nFrames,nCh] = size(kdata); 
data = zeros([N,N,nFrames]);
[nSamples,nInterleavesPerFrame,~] = size(kdata);

ktraj = reshape(ktraj/N,[nSamples,nInterleavesPerFrame,nFrames]);
kdata = reshape(kdata, [nSamples,nInterleavesPerFrame,nFrames,nCh]);
FT= NUFFT(ktraj,1,0,0,[N,N]);

Atb = zeros(N,N,nFrames);
for ii=1:nCh
    Atb = Atb + bsxfun(@times,FT'*kdata(:,:,:,ii),conj(csm(:,:,ii)));
end

ATA = @(x) SenseATA(x,FT,csm,N,nFrames,nCh) + lambda*x;
data = pcg(ATA,Atb(:),1e-3,50);
data = reshape(data,[N,N,nFrames]);

end

