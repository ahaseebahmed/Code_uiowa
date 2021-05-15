function data = l2Recont_encoder(kdata,FT,csm,lambda,N,W1,W2,b1,b2)
%function data = l2Recont(kdata,FT,csm,lambda,N)
%
% Arguments:
%   kdata   [np nv nf nc]   complex
%   FT      Fourier operator
%   csm     coil sens map
%   lambda      threshold
%   N               reconstucted image size
%
% Optional Arguments:

%
% Outputs:
%   data     [N N N Nt]      complex
%
% Ahmed, Abdul Haseeb <abdul-ahmed@uiowa.edu>

[~,~,nFrames,nCh] = size(kdata); 
Atb = zeros(N,N,nFrames);


for ii=1:nCh
    Atb = Atb + bsxfun(@times,FT'*kdata(:,:,:,ii),conj(csm(:,:,ii)));
end
lambda=max(abs(Atb(:)))*lambda;

Z=denoisers1(Atb,W1,W2,b1,b2);
ATA = @(x) SenseATA(x,FT,csm,N,nFrames,nCh) + lambda*(x-double(Z(:)));%Reg3(x);%+Reg1(x);%
data = pcg(ATA,Atb(:),1e-5,10);
data = reshape(data,[N,N,nFrames]);

end



function [X]= weighted_WT(x,N1,nf,WT)
X1=reshape(x,N1,N1,nf);
for i=1:nf
    X_WT=WT*X1(:,:,i);
    X_WT=reshape(X_WT(:)./((abs(X_WT(:)))+1e-10),[N1,N1]);
    X(:,:,i)=WT'*X_WT;
end
X=X(:);
end

function X=sth(x,N1,nf,WT,p)
X1=reshape(x,N1,N1,nf);
for i=1:nf
    X_WT=WT*X1(:,:,i);
    X_WT=(abs(X_WT)-p).*(abs(X_WT)>p);
    X(:,:,i)=WT'*X_WT;
end
X=X(:);
end

function X=l1_norm_WT(x,N1,nf,WT)
X1=reshape(x,N1,N1,nf);
for i=1:nf
        x1=WT*X1(:,:,i);
X(:,:,i)=WT'*(x1.*(x1.*conj(x1)+0.00001).^(-0.5));
end
X=X(:);
end

function X=l1_norm_TV(x,N1,nf)
X1=reshape(x,N1,N1,nf);
%for i=1:nf
       % [x1,x2]=gradient(X1);
         x1 = X1([2:end,end],:,:) - X1;
         x2 = X1(:,[2:end,end],:) - X1;
         x3 = X1(:,:,[2:end,end]) - X1;

         x1=(x1.*(x1.*conj(x1)+1e-10).^(-0.5));
         x2=(x2.*(x2.*conj(x2)+1e-10).^(-0.5));
         x3=(x3.*(x3.*conj(x3)+1e-10).^(-0.5));

        
        %res = cat(3,Dx,Dy);
%          parfor i=1:nf
%          X(:,:,i)=divergence(x1(:,:,i),x2(:,:,i));
%          end
%X=adjDx(x1)+adjDy(x2)+adjDz(x3);
X=adjDz(x3);
X=X(:);
end
function X=l1_norm_TVTemp(x,N1,nf)

X1=reshape(x,N1*N1,nf);
L = circulant([1,-1,zeros(1,nf-2)]);
X=X1*L;
%x1=fft(X1,[],3);
%x1=(x1.*(x1.*conj(x1)+1e-10).^(-0.5));
% x2=(x2.*(x2.*conj(x2)+1e-10).^(-0.5));
        
        %res = cat(3,Dx,Dy);
%          parfor i=1:nf
%          X(:,:,i)=divergence(x1(:,:,i),x2(:,:,i));
%          end
% X=adjDx(x1)+adjDy(x2);
%X=ifft(x1,[],3);
X=X(:);
end


function [X]= weighted_TV(x,N1,nf)
X1=reshape(x,N1,N1,nf);
    [x1,x2]=gradient(X1);
    x1=reshape(x1(:)./(2*abs(x1(:))+0.00001),[N1,N1,nf]);
    x2=reshape(x2(:)./(2*abs(x2(:))+0.00001),[N1,N1,nf]);

parfor i=1:nf
    X(:,:,i)=divergence(x1(:,:,i),x2(:,:,i));
end
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