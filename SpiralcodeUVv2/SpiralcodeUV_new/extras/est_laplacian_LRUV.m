
function [K, X, A] = est_laplacian_LRUV(nav,ktraj,kdata,csm_lowRes,N1, sigSq, lambda1)

nf = size(nav,2);
nav=double(nav);

p = 1;
q = 1-p/2;
eta = 2;
nBasis=100;
useGPU=1;

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
lambda2=0;
factor=0;
%N1=size(csm_lowRes,1);
X=zeros(N1*N1,nf);
[V,S,~] = svd(K);
W = V*((S+gamma*eye(nf))^(-q))*V';
A = W.*K;
A = -diag(sum(A))+A;
[~,Sbasis,V1]=svd(A);
V1=V1(:,1:nBasis);
Sbasis=Sbasis(1:nBasis,1:nBasis);

for i=1:1
  
   i
   
    Xpre=X;
    tic; x = solveUV(ktraj,kdata,csm_lowRes, V1,N1, 40,lambda1*Sbasis,useGPU,factor);toc
    X = (reshape(x,[N1*N1,nBasis])*V1');
    %Atb = Atb_LR(FT,kdata,csm_lowRes,true);
    %Reg = @(x) reshape(reshape(x,[N1*N1,nf])*(lambda1*A+lambda2*Lnn),[N1*N1*nf,1]);
    %AtA = @(x) AtA_LR(FT,x,csm_lowRes,nf,N1) + Reg(x);
    %tic; x1 = pcg(AtA,Atb(:),9e-4,40,[],[],Xpre(:));toc;%dataset virg 9e-4;
    %X =(reshape(x1,[N1*N1,nf]));
 
    %X = X*inv(eye(nf) + lambda*X);
%      X = nav*(eye(nf) + lambda*A);
   
    gamma = gamma/eta;
    
    X2 = sum(X.*conj(X),1);
    X3 = (X')*X;
    dsq = abs(repmat(X2,nf,1)+repmat(X2',1,nf)-2*real(X3));
    K = exp(-dsq/sigSq);
    [V,S,~] = svd(K);
    W = V*((S+gamma*eye(nf))^(-q))*V';
    A = W.*K;
    A = -diag(sum(A))+A;
    [~,Sbasis,V1]=svd(A);
    V1=V1(:,end-nBasis+1:end);
    Sbasis=Sbasis(end-nBasis+1:end,end-nBasis+1:end);
  
%      if (norm(X(:)-Xpre(:))<1e-5) 
%      break; 
%      end
%      if mod(i,5)==0
%          save(strcat('tt2mp1',num2str(i),'.mat'),'X');
%      end
    
end

