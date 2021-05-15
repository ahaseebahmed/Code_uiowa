
function res = AtAgpuBlock_old(X,Q,csm,n,nf,nc)

X = reshape(X,n,n,nf);
res = zeros(n,n,nf);
res1 = zeros(n,n,nf);
% csm = gpuArray(csm);

% n1 = 73;
% n2 = nf/n1;

tmp = zeros(2*n,2*n,nf);

for l=1:nf
%     res1 = gpuArray.zeros(n,n,n1);
%     X1 = gpuArray(X(:,:,(l-1)*n1+1:l*n1));
%     Q1 = gpuArray(Q(:,:,(l-1)*n1+1:l*n1));
        
    for k=1:nc
        tmp(n/2+1:end-n/2,n/2+1:end-n/2,:) = bsxfun(@times,X,csm(:,:,k));
        tmp1 = ifft2((fft2(tmp)).*Q);       
        res1 = res1+bsxfun(@times,tmp1(n/2+1:end-n/2,n/2+1:end-n/2,:),conj(csm(:,:,k)));
        tmp(:)=0;
    end
    
   % res(:,:,(l-1)*n1+1:l*n1) = gather(res1);
end

res = res1(:);
