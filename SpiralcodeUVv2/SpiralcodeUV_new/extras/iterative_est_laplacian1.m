
function [ X, A] = iterative_est_laplacian1(A,FT,WT,kdata,csm_lowRes,N1, sigSq, lam)

 nf = size(kdata,3);
% nav=double(nav);
% 
% p = 1;
% q = 1-p/2;
eta = 2;
% 
% X2 = double(sum(nav.*conj(nav),1));
% X3 = (nav')*nav;
% dsq = abs(repmat(X2,nf,1)+repmat(X2',1,nf)-2*real(X3));
% 
% % Changes
% %============
% medvalue = median(dsq(:));
% maxvalue = medvalue;%./sigSq;
% nav = nav./sqrt(maxvalue);
% dsq = dsq./maxvalue;
% %============
% tt=dsq/sigSq;
% K = exp(-tt);
% W=(circshift(eye(nf),[0,1])+circshift(eye(nf),[1,0]));
% %K=max(K,t);
% %W=K-eye(nf);
% W=(W+W')/2;
% Lnn = (diag(sum(W,1))-W);
L = circulant([1,-1,zeros(1,nf-2)]);
Lnn = L'*L/2;
gamma = 100;
lambda2=0;
%N1=size(csm_lowRes,1);
Xpre=zeros(N1*N1,nf);
% [V,S,~] = svd(K);
% W = V*((S+gamma*eye(nf))^(-q))*V';
% A = W.*K;
% A = -diag(sum(A))+A;

  Atb = Atb_LR(FT,kdata,csm_lowRes,true);
  Atb=Atb./max(abs(Atb(:)));
      lambda1=lam*max(abs(Atb(:)));

for i=1:10
  
   i
   
    
  
    Reg1 = @(x) reshape(reshape(x,[N1*N1,nf])*(lambda1*A+lambda2*Lnn),[N1*N1*nf,1]);
    %Reg2= @(x) weighted_WT(abs(x),N1,nf,WT);
    %Reg2= @(x) sth((x),N1,nf,WT,0.01);

    %Reg2= @(x) weighted_TV(x,N1,nf);
    %Reg3= @(x) l1_norm_WT(x,N1,nf,WT);%%slight better
    Reg3= @(x) l1_norm_TV((x),N1,nf);
    if i==20
        AtA = @(x) AtA_LR(FT,x,csm_lowRes,nf,N1)+Reg1(x);%+0.02*Reg3(x);
    else
        AtA = @(x) AtA_LR(FT,x,csm_lowRes,nf,N1)+Reg1(x);
    end
    tic; [x1,~,~,it,res] = pcg(AtA,Atb(:),1e-6,60,[],[],Xpre(:));toc;%dataset virg 9e-4;
    X =(reshape(x1,[N1*N1,nf]));
%      if (norm(X(:)-Xpre(:))<1e-15) 
%      break; 
%      end 
    Xpre=reshape(X,[N1*N1,nf]);
   % save(strcat('r',num2str(i),'.mat'),'X');
    [~,~,A]=estimateLapKernelLR(X,sigSq,lambda1);
  %  gamma = 100;
%  for j=1:10    
%     
%     X2 = sum(X.*conj(X),1);
%     X3 = (X')*X;
%     dsq = abs(repmat(X2,nf,1)+repmat(X2',1,nf)-2*real(X3));
%     K = exp(-dsq/sigSq);
%     [V,S,~] = svd(K); 
%     W = V*((S+gamma*eye(nf))^(-q))*V';
%     A = W.*K;
%     A = -diag(sum(A))+A;
%     X = Xpre*inv(eye(nf) + (lam)*A);
     gamma = gamma/eta;   
%   
% end
       
end
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
         x1=(x1.*(x1.*conj(x1)+1e-10).^(-0.5));
         x2=(x2.*(x2.*conj(x2)+1e-10).^(-0.5));
        
        %res = cat(3,Dx,Dy);
%          parfor i=1:nf
%          X(:,:,i)=divergence(x1(:,:,i),x2(:,:,i));
%          end
X=adjDx(x1)+adjDy(x2);
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