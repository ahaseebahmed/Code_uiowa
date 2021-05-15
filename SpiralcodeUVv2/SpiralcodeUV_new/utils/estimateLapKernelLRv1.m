
function [K, Z, A] = estimateLapKernelLRv1(nav, sigSq, lambda)

[n,nf] = size(nav);
Y=zeros(n,nf);
Z=zeros(n,nf);
X=nav;

nav=double(nav);

p = 1;
q = 1-p/2;
eta = 2;

X2 = double(sum(nav.*conj(nav),1));
X3 = (nav')*nav;
dsq = abs(repmat(X2,nf,1)+repmat(X2',1,nf)-2*real(X3));

% Changes
%============
medvalue = median(dsq(:));
maxvalue = medvalue;%./sigSq;
nav = nav./sqrt(maxvalue);
dsq = dsq./maxvalue;
%============
tt=dsq/sigSq;
K = exp(-tt);

[V,S,~] = svd(K);
gamma = 100;
beta=0.1;
lam=0.1;
%-----------
% X22 = double(sum(nav_lr.*conj(nav_lr),1));
% X33 = (nav_lr')*nav_lr;
% dsq1 = abs(repmat(X22,nf,1)+repmat(X22',1,nf)-2*real(X33));
% 
% % Changes
% %============
% medvalue = median(dsq1(:));
% maxvalue = medvalue;%./sigSq;
% nav_lr = nav_lr./sqrt(maxvalue);
% dsq1 = dsq1./maxvalue;
% %============
% tt=dsq1/sigSq;
% K1 = exp(-tt);
% 
% [V1,S1,~] = svd(K1);
%-----------
while(1)   
    W = V*((S+gamma*eye(nf))^(-q))*V';
    %W1 = V1*((S1+gamma*eye(nf))^(-q))*V1';

    
    A = W.*K;
    A = -diag(sum(A))+A;
     %A1 = W1.*K1;
    %A1 = -diag(sum(A1))+A1;
    %A = diag(sum(A))-A;
    
    Zpre = Z;
    Z = beta*(X+Y/beta)*inv(lambda*A + beta*eye(nf));
  
    X = (nav + beta*(Z-Y/beta))/(1+beta);
    
    Y = Y+ beta*(X-Z);
    
    if norm(Zpre-Z)<1e-5 && norm(X-Z)< 1e-4
        break;
    end
     %X = nav*inv(eye(nf) + lambda*A);
%      X = nav*(eye(nf) + lambda*A);

    
    gamma = gamma/eta;
    
    X2 = sum(Z.*conj(Z),1);
    X3 = (Z')*Z;
    dsq = abs(repmat(X2,nf,1)+repmat(X2',1,nf)-2*real(X3));
    K = exp(-dsq/sigSq);
    
    [V,S,~] = svd(K);
    
end
