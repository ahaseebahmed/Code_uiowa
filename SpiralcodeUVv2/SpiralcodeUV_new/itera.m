xx=abs(lowres(:,:,65));
%x_f=fft2(xx);
%imagesc((abs(double(x_f))))
x_f=fftshift(fft2(fftshift(xx)));
%x_f=fft2(x_f);
%x_f=fftshift(fftshift((x_f),1),2);
x_fz=padarray(x_f,[118 118],1e-5,'both');
imagesc((abs(double(x_fz))))
x_fz=fftshift((x_fz));
imagesc((abs(double(x_fz))))
yy=ifft2(x_fz);
figure;imagesc((abs(double(yy))),[0 1.15e-4]);colormap gray; axis image; axis off;
yy=fftshift((yy));

%%
tmp1=fft2(fftshift(xx));
%tmp2=fftshift(tmp1);
tmp3= ifft2(tmp1,300,300);
imagesc(abs(xx))

