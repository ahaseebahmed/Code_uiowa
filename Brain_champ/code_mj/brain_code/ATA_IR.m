function out = ATA_IR(x,p)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

[nx,ny,nz] = size(p{1}.Atb);
nt = length(p);

x = reshape(x,[nx,ny,nz,nt]);

out = zeros(nx,ny,nz,nt);
for i=1:nt
    out(:,:,:,i) = p{i}.FT'*(p{i}.FT*x(:,:,:,i));
end

out = out(:);

