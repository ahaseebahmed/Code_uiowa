
function [W,D,L] = giveSmoothedGraph(X,N,alpha,beta)

[~, n] = size(X);


X2 = sum(X.*conj(X),1);
X3 = (X')*X;
d = abs(repmat(X2,n,1)+repmat(X2',1,n)-2*real(X3));
for i=1:n, d(i,i)=0; end
d = d./max(d(:));
    
W = gsp_learn_graph_log_degrees(d,1,alpha);
t = beta*(circshift(eye(n),[0 1]) + circshift(eye(n),[1 0]));
W = max(W,t);

    
G = gsp_graph(W);

G = gsp_estimate_lmax(G);
G = gsp_compute_fourier_basis(G);
D = G.U(:,1:N);
L = G.L;



