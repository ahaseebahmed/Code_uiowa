function test = biasCorrect(Volume2View,order)

[nx,ny,nz] = size(Volume2View);
[x,y,z] = meshgrid(1:nx,1:ny,1:nz);

Mat = [ones(size(x(:))),x(:),y(:),z(:)];
if(order>1)
    Mat = [Mat,x(:).^2,y(:).^2,z(:).^2,x(:).*y(:),x(:).*z(:),y(:).*z(:)];
end
Q = Mat'*Mat;
coeffs = pinv(Q)*Mat'*Volume2View(:);
biasfield = Mat*coeffs;
biasfield = reshape(biasfield,[nx,ny,nz]);
test = Volume2View./biasfield;
test = test./max(test(:));

end

