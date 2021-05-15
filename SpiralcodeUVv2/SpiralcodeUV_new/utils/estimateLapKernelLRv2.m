
function [K, Z, A] = estimateLapKernelLRv2(nav,nav_lr, sigSq, lambda)

[n,nf] = size(nav);
Y=zeros(n,nf);
Z=nav;
X=zeros(n,nf);

nav=double(nav);

p = 1;
q = 1-p/2;
eta = 2;

X2 = double(sum(nav_lr.*conj(nav_lr),1));
X3 = (nav_lr')*nav_lr;
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
i=0;
%-----------
while(1)   
    W = V*((S+gamma*eye(nf))^(-q))*V';
    
    A = W.*K;
    A = -diag(sum(A))+A;
    %A = diag(sum(A))-A;
    
    Zpre = Z;
    Z = beta*(X+Y/beta)*inv(lambda*A + beta*eye(nf));
  
    X = (nav + beta*(Z-Y/beta))/(1+beta);
    
    Y = Y+ beta*(X-Z);
    
    if norm(Zpre-Z)<1e-4 && norm(X-Z)< 1e-4
        break;
    end
     %X = nav*inv(eye(nf) + lambda*A);
%      X = nav*(eye(nf) + lambda*A);
if i>30
    i=0;
end
i=i+1;
    
    gamma = gamma/eta;
    
    X2 = sum(Z.*conj(Z),1);
    X3 = (Z')*Z;
    dsq = abs(repmat(X2,nf,1)+repmat(X2',1,nf)-2*real(X3));
    K = exp(-dsq/sigSq);
    
    [V,S,~] = svd(K);
    
end
