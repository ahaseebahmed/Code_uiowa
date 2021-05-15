function [K, X, A] = estimateLR(nav, sigSq, lambda)

nf = size(nav,2);
nav=double(nav);

p = 1;
q = 1-p/2;
eta = 2;

X2 = double(sum(nav.*conj(nav),1));
X3 = (nav')*nav;
dsq = abs(repmat(X2,nf,1)+repmat(X2',1,nf)-2*real(X3));

dsq = dsq - diag(diag(dsq));

K = exp(-dsq/(sigSq));
[V,S,~] = svd(K);
s = sort(diag(S),'descend');
%============


[V,S,~] = svd(K);
gamma = 100;

for i=1:70    
    W = V*((S+gamma*eye(nf))^(-q))*V';
    
    A = W.*K;
    A = -diag(sum(A))+A;
    
    X = nav*inv(eye(nf) + (lambda)*A);
    
    gamma = gamma/eta;
    
    X2 = sum(X.*conj(X),1);
    X3 = (X')*X;
    dsq = abs(repmat(X2,nf,1)+repmat(X2',1,nf)-2*real(X3));
    dsq = dsq - diag(diag(dsq));
    K = exp(-dsq/(sigSq));
    
    [V,S,~] = svd(K);
    
end