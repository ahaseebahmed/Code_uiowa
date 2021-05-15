
function [K, X, A] = estimateLapKernelLR_reg(nav, sigSq, lambda)

nf = size(nav,2);
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
t=0.87*(circshift(eye(nf),[0,1])+circshift(eye(nf),[1,0]));
K=max(K,t);

[V,S,~] = svd(K);
gamma = 100;

for i=1:70    
    W = V*((S+gamma*eye(nf))^(-q))*V';
    
    A = W.*K;
    %t=0.05*(circshift(eye(nf),[0,1])+circshift(eye(nf),[1,0]));
    %A=max(A,t);
    %A=A+t;
    A = -diag(sum(A))+A;
    
     X = nav*inv(eye(nf) + (lambda/sqrt(i))*A);
%      X = nav*(eye(nf) + lambda*A);

    
    gamma = gamma/eta;
    
    X2 = sum(X.*conj(X),1);
    X3 = (X')*X;
    dsq = abs(repmat(X2,nf,1)+repmat(X2',1,nf)-2*real(X3));
    K = exp(-dsq/sigSq);
    [V,S,~] = svd(K);
    
end

