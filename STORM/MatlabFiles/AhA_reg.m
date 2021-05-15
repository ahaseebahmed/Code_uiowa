
function res = AhA_reg(U,S,L,csm,n,nf,nc,LamMtx,WeightL1,WeightFourier,useGPU,gw,W1,W2,b1,b2)

U       = reshape(U,n,n,nf);
res     = zeros([n,n,nf],'double');
tmp3    = zeros([n^2,1],'double');
    
%% GPU implementation
if(useGPU) 
    U = gpuArray(U);
    res = gpuArray(res);
    tmp3 = gpuArray(tmp3);
for i=1:nf
    for k=1:nc    
        tmp1 = (fft2(bsxfun(@times,U(:,:,i),csm(:,:,k))));
        tmp3(S{i})=tmp1(S{i});
        res(:,:,i) = res(:,:,i)+bsxfun(@times,ifft2(reshape(tmp3,n,n)),conj(csm(:,:,k)));
        tmp3(:)=0;
    end
end

%     if(nargin>9)
%         if(~isempty(WeightL1))
%            WeightL1 = reshape(WeightL1,n,n,r);
%           if(useGPU)
%                WeightL1 = gpuArray(WeightL1);
%           end
%           U = U.*WeightL1;
%         end
%     end
    
    %reg_term = reshape(U,n^2,r);
%     if(nargin>12)
%         reg_term(:,1:end-1) = reg_term(:,1:end-1).*reshape(ww,[n^2 r-1]);
%     end
     %reg_term = reshape(U,n^2,nf)*L;

    gw=1+10*gw;
    gw = repmat(gw,[1,1,nf]);
    %U=gw.*U;
    reg_term = denoisers1(U,W1,W2,b1,b2,gw);
    reg_term=reshape(reg_term,[n,n,nf]);
    %gw=gw-1;
    %tmpp=mean(reg_term,3);
    %gw=gw.*tmpp;
    %gw=gw+1;
    %reg_term=reg_term.*gw;
    %reg_term2=gw.*U;
    res = res(:) + 0.05*reg_term(:);%+reg_term2(:);
    fprintf('reg_term %f \n',reg_term(:)'*reg_term(:));

%     if(nargin>10 && ~isempty(WeightFourier))
%         tmp1 = ifft2((fft2(U).*WeightFourier));
%         res = res(:) + tmp1(:);
%     end

else
%% CPU Option

     for k=1:nc    
        tmp1 = reshape(fft2(bsxfun(@times,U,csm(:,:,k))),n^2,r);
    
        for i=1:nf       
        tmp3(S{i},:) = tmp3(S{i},:) + (tmp1(S{i},:)*V(:,i))*V(:,i)';
        end
    
        res = res+bsxfun(@times,ifft2(reshape(tmp3,n,n,r)),conj(csm(:,:,k)));
        tmp3(:)=0;
    end

    if(nargin>9)
        if(~isempty(WeightL1))
           LamMtx = double(LamMtx);
           WeightL1 = reshape(WeightL1,n,n,r);
          if(useGPU)
               WeightL1 = gpuArray(WeightL1);
               LamMtx = gpuArray(LamMtx);
          end
          U = U.*WeightL1;
        end
    end

    reg_term = reshape(U,n^2,r)*LamMtx;
    res = res(:) + reg_term(:)+0.01*l1_norm_TV(U,V,n,r);

    if(nargin>10 && ~isempty(WeightFourier))
        tmp1 = ifft2((fft2(U).*WeightFourier));
        res = res(:) + tmp1(:);
    end
end
end

function X=l1_norm_TV(u,v,n,r)
u=reshape(u,n*n,r);
nf=size(v,2);
X1=u*v;
X1=reshape(X1,[n,n,nf]);
x3 = X1(:,:,[2:end,end]) - X1;
x3=(x3.*(x3.*conj(x3)+1e-10).^(-0.5));
X=adjDz(x3);
X=reshape(X,[n*n,nf]);
X=X*v';
X=X(:);
end

function res = adjDz(x)
res= x(:,:,[1,1:end-1]) - x;
res(:,:,1) = -x(:,:,1);
res(:,:,end) = x(:,:,end-1);
end

