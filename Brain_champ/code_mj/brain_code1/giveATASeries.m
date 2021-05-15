function out = giveATASeries(x,FT,wts,N)

x = reshape(x,N);
out = zeros(N);

for i=1:N(end),
    dtemp = FT*x(:,:,:,i);
    out(:,:,:,i) = FT'*bsxfun(@times,dtemp,wts(:,i));
end
out = out(:);

end

