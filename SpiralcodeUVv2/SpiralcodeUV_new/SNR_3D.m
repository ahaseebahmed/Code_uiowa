function SNR=SNR_3D(u0,u)
%  u0: orginal image
%  u: noised image
%  (nb,na): size
if size(u0)~=size(u)
    disp('They are not the same size')
    exit
end

%[nb,na,nc]=size(u0);
MSE=norm(abs(u(:)-u0(:)),'fro')^2./(201*201);
SNR=10.*log10(max(u0(:))/MSE);
%SNR=10.*log10(norm(u0(:),'fro')^2/MSE);

return