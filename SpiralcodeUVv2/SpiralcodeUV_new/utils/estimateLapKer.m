
function [K, X, A] = estimateLapKer(nav,lambda)

nf = size(nav,2);
nav=double(nav);
p = 1;
q = 1-p/2;
eta = 2;
gamma = 100;

K = abs(nav'*nav);
[V,S,~] = svd(K);

for i=1:70    
    W = V*((S+gamma*eye(nf))^(-q))*V';
    Q=sqrt(W);
    %A=W+W';
    A=Q*Q';
    X = nav*inv(eye(nf) + lambda*A);
    gamma = gamma/eta;
    K = abs(X'*X);
    [V,S,~] = svd(K);   
end

