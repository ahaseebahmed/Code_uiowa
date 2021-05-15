function y = AtA_UV(FT,x,V,csm,N,NsamplesPerFrame,w)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
%csm=ones(N,N,N,1);
[~,~,~,nChannelsToChoose] = size(csm);
[nFrames,nBasis] = size(V);
x = reshape(x,[N,N,N,nBasis]);

y = zeros(N,N,N,nBasis);

for j=1:nChannelsToChoose
    virtualKspace = zeros(NsamplesPerFrame,nFrames);
    for i=1:nBasis
        temp = FT*(x(:,:,:,i).*csm(:,:,:,j));
        virtualKspace = virtualKspace + reshape(temp,[NsamplesPerFrame,nFrames])*diag(V(:,i));
    end
    virtualKspace=virtualKspace.*w;
    for i=1:nBasis
        temp = virtualKspace*diag(V(:,i));
        y(:,:,:,i) = y(:,:,:,i) + (FT'*(temp(:))).*conj(csm(:,:,:,j));  
    end
end

%reg=reshape(y,[N*N*N,nBasis])*Sbasis*0.1;
y = y(:);%+reg(:);

end

