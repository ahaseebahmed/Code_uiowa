function [data,Atb] = l2Recont_v3(kdata,FT,csm,lambda,N)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here

[~,~,nFrames,nCh] = size(kdata); 
%N=size(csm,1);

%w=repmat(dcf(1:nSamples),[1 nInterleavesPerFrame]);

Atb = zeros(N,N,nFrames);
for ii=1:nCh
    Atb = Atb + bsxfun(@times,FT'*kdata(:,:,:,ii),conj(csm(:,:,ii)));
end
lambda=max(abs(Atb(:)))*lambda;

ATA = @(x) SenseATA(x,FT,csm,N,nFrames,nCh) + lambda*x;
%ATA = @(x) SenseATA(x,FT,csm,N,nFrames,nCh) + lambda*x.*(x.*x+eps).^(-0.5);
%ATA = @(x) SenseATA(x,FT,csm,N,nFrames,nCh) + lambda*XL(x,L,N,nFrames);
data = pcg(ATA,Atb(:),1e-5,50);
data = reshape(data,[N,N,nFrames]);
% save('LR_wo_storm1.mat','data');

end

function res=XL(x,L,N,nFrames)
x=reshape(x,[N*N nFrames]);
res=x*L;
res=res(:);
end
