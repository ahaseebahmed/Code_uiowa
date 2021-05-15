
function [K, X] = est_LRKer1(nav,FT,kdata,csm_lowRes,N1,lambda)

nf = size(nav,2);
nav=double(nav);
p = 1;
q = 1-p/2;
eta = 2;
gamma = 100;

K = abs(nav'*nav);
[V,S,~] = svd(K);
X=nav;

for i=1:10    
    W = V*((S+gamma*eye(nf))^(-q))*V';
    
     Atb = Atb_LR(FT,kdata,csm_lowRes,true);
    %Reg = @(x) reshape(reshape(x,[N1*N1,nf])*(lambda1*A),[N1*N1*nf,1]);
    AtA = AtA_LR(FT,X,csm_lowRes,nf,N1);
    X = X-lambda*(X*W+(reshape(AtA,[N1*N1,nf])-reshape(Atb,[N1*N1,nf])));
    
    gamma = gamma/eta;
    K = abs(X'*X);
    [V,S,~] = svd(K);   
end

