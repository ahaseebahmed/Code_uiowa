
function [K, X, A] = LRrecon(nav, sigSq, lambda)

[n,nf] = size(nav);
Y=zeros(n,nf);
Z=zeros(n,nf);
X=nav;

nav=double(nav);

p = 1;
q = 1-p/2;
eta = 2;

gamma = 100;
beta=0.1;
lam=0.1;

while(1)
    
    Zpre=Z;
    [U,St,V]=svd(X+Y/beta,0);
    lam=St(1)*0.2;
    x=diag(St);
    St=diag((abs(x)-lam).*x./abs(x).*(abs(x)>lam));
    St(isnan(St))=0;
    Z=U*St*V';
    X=(nav+beta*Z-Y)*(1+beta)^-1;
    Y=Y+beta*(X-Z);
    
   
    if norm(Zpre-Z)<1e-10
        break;
    end
   
    
end
