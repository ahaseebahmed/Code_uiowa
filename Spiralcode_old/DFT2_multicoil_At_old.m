function res=DFT2_multicoil_At_old(x,At_DFT,CSM,n,nf,nc)

res = zeros(n,n,nf);

for i=1:nf
    for j=1:nc
        idata_t=double(At_DFT{i}(x(:,i,j)));
        res(:,:,i) = res(:,:,i) + conj(CSM(:,:,j)).*idata_t;
    end
end

res = res(:);












