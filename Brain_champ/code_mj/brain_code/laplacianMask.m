function Rxy = laplacianMask(x,mask)

[n1,n2,n3] = size(mask);
x = reshape(x,[n1,n2,n3]);
x = bsxfun(@times,x,mask); 

g1 = x - circshift(x,[1,0,0]);
g1 = bsxfun(@times,g1,mask==circshift(mask,[1,0,0]));
Rxy = g1- circshift(g1,[-1,0,0]);

g1 = x - circshift(x,[0,1,0]);
g1 = bsxfun(@times,g1,mask==circshift(mask,[0,1,0]));
Rxy = Rxy + g1 - circshift(g1,[0,-1,0]);

g1 = x - circshift(x,[0,0,1]);
g1 = bsxfun(@times,g1,mask==circshift(mask,[0,0,1]));
Rxy = Rxy + g1 - circshift(g1,[0,0,-1]);

Rxy = Rxy(:);
