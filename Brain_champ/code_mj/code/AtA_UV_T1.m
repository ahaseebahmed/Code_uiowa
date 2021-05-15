function y = AtA_UV_T1(FT,x,Vbasis,dcf,N,szdata)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

[nt,nBasis] = size(Vbasis);
x = reshape(x,[N,N,N,nBasis]);
nvCh = szdata(1);
npts = szdata(2);
nsamples = npts/nt;

kdata = zeros([nvCh*nsamples,nt]);
for i=1:nBasis
     temp = FT*x(:,:,:,i);
     kdata = kdata + reshape(temp',[nvCh*nsamples,nt])*diag(Vbasis(:,i));
end

y = zeros(size(x));
for i=1:nBasis
    dtemp = kdata*diag(Vbasis(:,i));
    dtemp = reshape(dtemp,[nvCh,npts]);
    y(:,:,:,i) = FT'*(bsxfun(@times,dtemp,dcf')');
end

y = y(:);

end

