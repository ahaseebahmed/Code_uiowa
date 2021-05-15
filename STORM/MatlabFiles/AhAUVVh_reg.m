
function res = AhAUVVh_reg(U,V,S,csm,n,r,nf,nc,LamMtx,WeightL1,WeightFourier,useGPU,ww)

U       = reshape(U,n,n,r);
res     = zeros([n,n,r],'double');
tmp3    = zeros([n^2,r],'double');
%useGPU=1;   
%% GPU implementation
if(useGPU) 
    U = gpuArray(U);
    res = gpuArray(res);
    tmp3 = gpuArray(tmp3);

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
           WeightL1 = reshape(WeightL1,n,n,r);
          if(useGPU)
               WeightL1 = gpuArray(WeightL1);
          end
          U = U.*WeightL1;
        end
    end
    
    reg_term = reshape(U,n^2,r);
    if(nargin>12)
        reg_term(:,1:end-1) = reg_term(:,1:end-1).*reshape(ww,[n^2 r-1]);
    end
     reg_term = reshape(reg_term,n^2,r)*LamMtx;
    res = res(:) + reg_term(:);%+0.0005*l1_norm_TV(U,V,n,r);

    if(nargin>10 && ~isempty(WeightFourier))
        tmp1 = ifft2((fft2(U).*WeightFourier));
        res = res(:) + tmp1(:);
    end

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

% tmp3=zeors(n^2,r);
%     for k=1:nc    
%         tmp1 = reshape(fft2(bsxfun(@times,res,csm(:,:,k))),n^2,r);
%         tmp1=reshape(tmp1,[n,n,r]);
%         for i=1:nf 
%         tmp2 = bsxfun(@times,tmp1,S(:,:,i));
%         tmp2=reshape(tmp2,[n^2,r]);
%         %tmp2 = bsxfun(@times,tmp1,S(:,:,i));
%         tmp3 = tmp3 + (tmp2*V(:,i))*V(:,i)';
%         end
%     
%         res = res+bsxfun(@times,ifft2(reshape(tmp3,n,n,r)),conj(csm(:,:,k)));
%         tmp3(:)=0;
%     end
%     
%         for k in range(NF):
%             tmp=maskT[k,:,:,:]
%             tmp=tmp.repeat(nbasis,1,1,1).to(gpu)
%             tmp2=tmp*tmp2
%             tmp2=tmp2.to(gpu,dtype)
%             tmp3=VT[:,k].unsqueeze(1)
%             tmp4=torch.matmul(tmp2.permute(1,2,3,0),tmp3)
%             tmp4=torch.matmul(tmp4,tmp3.T)        
%             tmp4=tmp4.permute(3,0,1,2)
%             tmp5=tmp5+tmp4*tmp.to(gpu)
%      
%         del tmp,tmp2,tmp3,tmp4   
%         tmp1=sf.pt_ifft2c(tmp5)
%         tmp=csmConj[i,:,:,:]
%         tmp=tmp.repeat(nbasis,1,1,1).to(gpu)
%         tmp2=sf.pt_cpx_multipy(tmp,tmp1)
%         atbv=atbv+tmp2
%         tmp5=torch.zeros(nbasis,nx,nx,2)
%         tmp5=tmp5.to(gpu)