function [Atb,Q] = giveOperators(kdata,ktraj,N,dcf)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here

[nSamples,nInterleavesPerFrame,Nframes,nCh] = size(kdata);



ktraj = reshape(ktraj/N,[nSamples,nInterleavesPerFrame,Nframes]);

FT= NUFFT(ktraj,dcf,0,0,[N,N]);


Atb = zeros(N,N,Nframes,nCh);
for ii=1:nCh
    Atb(:,:,:,ii) = Atb(:,:,:,ii) + FT'*double(kdata(:,:,:,ii));
end

FT= NUFFT(ktraj,1,0,0,[2*N,2*N]);
Q = fftshift(fftshift(FT'*ones(size(ktraj)),1),2);


end

