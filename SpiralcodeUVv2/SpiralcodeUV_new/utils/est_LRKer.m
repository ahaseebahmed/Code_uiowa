
function [K, X, A] = est_LRKer(nav,lambda)

nf = size(nav,2);
nav=double(nav);
p = 1;
q = 1-p/2;
eta = 2;
gamma = 100;

K = abs(nav'*nav);
[V,S,~] = svd(K);
X=nav;

for i=1:100    
    W = V*((S+gamma*eye(nf))^(-q))*V';
    %Q=sqrt(W);
    %A=W+W';
    A=W;
    %A=Q+Q';
    %A=Q;
    X = nav*inv(eye(nf) + lambda*A);
    
    gamma = gamma/eta;
    K = abs(X'*X);
    [V,S,~] = svd(K);   
end

