load('bCom.mat')
n=512;
nf = 1000;
bCom = reshape(bCom,n,4,24,nf);

[y,x] = meshgrid(-nf/2:nf/2-1,-n/2:n/2-1);
sigma_x = n/4;
sigma_y = nf/2;
filterx = exp(-x.^2./(sigma_x.^2));
filtery = exp(-y.^2./(sigma_y.^2));
%filter = reshapeim(filterx,[n,1,1,nf]);
filter = reshape(filterx.*filtery,[n,1,1,nf]);

filter = repmat(filter,[1,4,24,1]);
filter = 0*filter+1;

 
bComFiltered = fftshift(fft(bCom,[],4),4);
bComFiltered = bComFiltered.*filter;
bComFiltered = ifft(ifftshift(bComFiltered,4),[],4);

B1 = (fftshift(ifft(fftshift(bComFiltered,1),[],1),1));
B2 = reshape(B1,512*4,24,1000);

Bnew = squeeze(sqrt(sum(abs(B2).^2,2)));