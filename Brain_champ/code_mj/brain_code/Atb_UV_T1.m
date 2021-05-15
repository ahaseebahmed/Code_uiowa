function Atb = Atb_UV_T1(FT,kdata,Vbasis,dcf,N)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

[nt,nBasis] = size(Vbasis);
Atb = zeros(N,N,N,nBasis);
[nvCh,npts] = size(kdata);
nsamples = npts/nt;

for i=1:nBasis
    dtemp = reshape(kdata,nvCh*nsamples,nt)*diag(Vbasis(:,i));
    dtemp = reshape(dtemp,[nvCh,npts]);
    Atb(:,:,:,i) = FT'*(bsxfun(@times,dtemp,dcf')');
end

