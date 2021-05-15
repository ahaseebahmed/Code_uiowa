function X=l2_norm_TV(X1,N1)

x1 = X1([2:end,end],:,:) - X1;
x2 = X1(:,[2:end,end],:) - X1;
x3 = X1(:,:,[2:end,end]) - X1;

% x1=(x1.*(x1.*conj(x1)+1e-10).^(-0.5));
% x2=(x2.*(x2.*conj(x2)+1e-10).^(-0.5));
% x3=(x3.*(x3.*conj(x3)+1e-10).^(-0.5));

        
        %res = cat(3,Dx,Dy);
%          parfor i=1:nf
%          X(:,:,i)=divergence(x1(:,:,i),x2(:,:,i));
%          end
X=adjDx(x1)+adjDy(x2)+adjDz(x3);
%X=adjDz(x3);
X=X(:);
end


function res = adjDy(x)
res = x(:,[1,1:end-1],:) - x;
res(:,1,:) = -x(:,1,:);
res(:,end,:) = x(:,end-1,:);
end
function res = adjDx(x)
res = x([1,1:end-1],:,:) - x;
res(1,:,:) = -x(1,:,:);
res(end,:,:) = x(end-1,:,:);
end
function res = adjDz(x)
res= x(:,:,[1,1:end-1]) - x;
res(:,:,1) = -x(:,:,1);
res(:,:,end) = x(:,:,end-1);
end