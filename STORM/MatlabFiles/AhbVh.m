
function res = AhbVh(b,S,V,conjcsm,n,r,nf,nc,useGPU)

res = zeros(n^2,r);
tmp = zeros(n^2,1);
tmp2 = zeros(n,n);

if(useGPU)
    V = gpuArray(V);
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
    res = res+reshape(tmp2,n^2,1)*V(:,i)';
    tmp2(:) = 0;
end

res = n*res(:);



% for i=1:400
%     tmp2=atb(:,:,i);
%     res = res+reshape(tmp2,n^2,1)*V(:,i)';
% end