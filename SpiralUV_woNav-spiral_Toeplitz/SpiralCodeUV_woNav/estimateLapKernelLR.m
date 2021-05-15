function [A,maxvalue] = estimateLapKernelLR(nav, sigSq)

nf = size(nav,2);
nav=double(nav);

p = 1;
q = 1-p/2;
eta = 1;

X2 = double(sum(abs(nav).^2,1));
X3 = (nav')*nav;
dsq = abs(repmat(X2,nf,1)+repmat(X2',1,nf)-2*real(X3));
dsq = dsq - diag(diag(dsq));
neighbors = diag(dsq,1);
neigbors = neighbors(20:end-20);
maxvalue = max(neigbors);
K = exp(-dsq/(sigSq));
[V,S,~] = svd(K);

s = (diag(S)+eta).^(-q);
W = V*diag(s)*V';
    
%A = K;
A =  W.*K;
A = diag(sum(A))-A;
if(mean(diag(A))<0)
    A = -A;
end
end