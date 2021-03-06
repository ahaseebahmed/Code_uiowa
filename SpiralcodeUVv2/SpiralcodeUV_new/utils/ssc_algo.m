
function [A] = ssc_algo(nav, lam1, rho)

nf = size(nav,2);
sigSq=2.5;
nav=double(nav);

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
A=zeros(nf,nf);%K;
C=zeros(nf,nf);
d=zeros(nf,1);
delta=zeros(nf,nf);


for i=1:25    
    Apre=A;
    A=inv(lam1*K+rho*eye(nf)+rho*ones(nf,nf))*(lam1*K+rho*(ones(nf,nf)+C-diag(C))-ones(nf,1)*d'-delta);
    J=soft_thres(A+delta./rho,1/rho);
    C=J-diag(J);
    d=d+rho*(A'*ones(nf,1)-ones(nf,1));
    delta=delta+rho*(A-C+diag(C));
    
    if norm(A'*ones(nf,1)-ones(nf,1))<1e-5 && norm(A-C)< 1e-5
        break;
    end
    fprintf('%12.8f \n',norm(Apre-A));
    
end
end

function res=soft_thres(X,lam)
    
nff=size(X,2);    
res=abs(X(:))>0.*(abs(X(:))-lam);
res=reshape(res,[nff,nff]);

end

