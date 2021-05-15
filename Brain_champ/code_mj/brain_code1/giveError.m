function error = giveError(x,p)


x
[k1_new,nS] = givePerturbedTraj(x(1),x(2),p);  

FT = gpuNUFFT(transpose(k1_new),ones(size(p.dcf_new)),p.osf,p.wg,p.sw,[1 1 1]*p.mtx_reco,[],true);
image = FT'*bsxfun(@times,p.q,p.dcf_new);

s = transpose(FT*image);
s = reshape(s,1,nS,p.nangles);
for i=1:1140:p.nangles,
    s(:,:,i:i+2)=0;
end
s = reshape(s,1,nS*p.nangles);

error = norm(abs(p.q(:)-s(:)))/norm(abs(p.q(:)));
end

