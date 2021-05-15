
function res = Ahb(b,S,conjcsm,n,nf,nc,useGPU)

res = zeros(n,n,nf);
tmp = zeros(n^2,1);
tmp2 = zeros(n,n);

if(useGPU)
    res = gpuArray(res);
    tmp = gpuArray(tmp);
    tmp2 = gpuArray(tmp2);
end

for i=1:nf
    for k=1:nc
        tmp(S{i})= b{i,k};
        tmp=reshape(tmp,n,n);
        tmp2 = tmp2+conjcsm(:,:,k).*ifft2(tmp);
        tmp(:) = 0;
    end
    res(:,:,i)=tmp2;
    tmp2(:) = 0;
end

res = n*res(:);
