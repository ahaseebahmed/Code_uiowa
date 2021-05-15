
function [K, X, A] = est_laplacian_LR1(nav,FT,kdata,csm_lowRes, sigSq, lambda1)

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
W=(circshift(eye(nf),[0,1])+circshift(eye(nf),[1,0]));
%K=max(K,t);
%W=K-eye(nf);
W=(W+W')/2;
Lnn = (diag(sum(W,1))-W);
gamma = 100;
lambda2=0.005;
N1=size(csm_lowRes,1);
A=zeros(nf,nf);


for i=1:50    
      if mod(i,5)==0
    Atb = Atb_LR(FT,kdata,csm_lowRes,true);
    Reg = @(x) reshape(reshape(x,[N1*N1,nf])*(lambda1*A+lambda2*Lnn),[N1*N1*nf,1]);
    AtA = @(x) AtA_LR(FT,x,csm_lowRes,nf) + Reg(x);
    tic; x1 = pcg(AtA,Atb(:),1e-5,50);toc;
    X =(reshape(x1,[N1*N1,nf]));
      else
    X = nav*inv(eye(nf) + lambda1*A);
%      X = nav*(eye(nf) + lambda*A);
      end
    gamma = gamma/eta;
    
    X2 = sum(X.*conj(X),1);
    X3 = (X')*X;
    dsq = abs(repmat(X2,nf,1)+repmat(X2',1,nf)-2*real(X3));
    K = exp(-dsq/sigSq);
    [V,S,~] = svd(K);
    W = V*((S+gamma*eye(nf))^(-q))*V';
    A = W.*K;
    A = -diag(sum(A))+A;
  
    
end

