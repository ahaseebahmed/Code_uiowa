
function [K, X] = est_LR3(nav,FT,kdata,csm_lowRes,N1, sigSq, lambda)

nf = size(nav,2);
nav=double(nav);
p = 1;
q = 1-p/2;
eta = 2;
gamma = 100;

% K = abs(nav'*nav);
% [V,S,~] = svd(K);

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
% gamma = 100;
% lambda2=0.9;
% %N1=size(csm_lowRes,1);
% X=zeros(N1*N1,nf);
% [V,S,~] = svd(K);
% W = V*((S+gamma*eye(nf))^(-q))*V';
% A = W.*K;
% A = -diag(sum(A))+A;

Y=nav;Z=0;
idd=eye(N1,N1);
idd=repmat(idd,[1,1,nf]);
for i=1:10
  
   i
   
%     K = abs(X'*X);
%     [V,S,~] = svd(K); 
%     W = V*((S+gamma*eye(nf))^(-q))*V';
%     X = X*(eye(nf) - lambda*W);
    %Q=sqrt(W);
    %A=W+W';
    %A=Q*Q';
    [U,S,V]=svd(Y-Z,0);
    S_th=diag(soft_thres(diag(S),S(1)*lambda));
    L=U*S_th*V';
    
    Atb = Atb_LR(FT,kdata,csm_lowRes,true);
    Atb=reshape(Atb,[N1*N1,nf]);
    %Reg = @(x) reshape(reshape(x,[N1*N1,nf])*(lambda1*A),[N1*N1*nf,1]);
    AtA = inv(AtA_LR(FT,idd,csm_lowRes,nf,N1)+0.1*idd);
    AtA=reshape(AtA,[N1*N1,nf]);
    Y=*(Atb-(L+Z));
    %tic; x1 = pcg(AtA,Atb(:),1e-4,10,[],[],Atb(:));toc;%dataset virg 9e-4;
    Z=Z+0.1*(L-Y);
    %(reshape(x1,[N1*N1,nf]));
   
    
end
end

function res = soft_thres(x,p)
res=(abs(x)-p).*(abs(x)>p);
res(isnan(res))=0;
end