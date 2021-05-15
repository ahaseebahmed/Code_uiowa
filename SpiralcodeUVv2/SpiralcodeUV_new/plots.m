for i=70:150;colormap gray;pause(0.1)
subplot(211);imagesc(abs(squeeze(x(:,i,:))));
subplot(212);imagesc(abs(squeeze(y(:,i,:))));
end


xx=abs(squeeze(x(:,35:192,:)));
yy=abs(squeeze(y(29:228,63:220,:)));
 xx=(xx-min(xx(:)))/(max(xx(:))-min(xx(:)));
 yy=(yy-min(yy(:)))/(max(yy(:))-min(yy(:)));
xx=rot90(abs(xx),-1);
yy=rot90(abs(yy),-1);
 
 
 
for i=1:1;colormap gray;pause(0.1)
subplot(311);imagesc(xx(:,:,i));
subplot(312);imagesc(yy(:,:,i));
subplot(313);imagesc(abs(yy(:,:,i)-xx(:,:,i)));
end

SNR_3D(yy,xx)



xx=abs(squeeze(xx(28:90,106:154,:)));
yy=abs(squeeze(yy(28:90,107:155,:)));
 xx=(xx-min(xx(:)))/(max(xx(:))-min(xx(:)));
 yy=(yy-min(yy(:)))/(max(yy(:))-min(yy(:)));

for i=1:1;colormap gray;pause(0.1)
subplot(311);imagesc(xx(:,:,i));
subplot(312);imagesc(yy(:,:,i));
subplot(313);imagesc(abs(yy(:,:,i)-xx(:,:,i)));
end

norm(xx(:)-yy(:))/norm(xx(:))
for i=1:400;pause(0.1); colormap gray;
    subplot(221);imagesc(fliplr(flipud(abs(y1(:,:,i)))));
    subplot(222);imagesc(fliplr(flipud(abs(y2(:,:,i)))));
    subplot(223);imagesc(fliplr(flipud(abs(y3(:,:,i)))));
    subplot(224);imagesc(fliplr(flipud(abs(y(:,:,i)))));
end

for i=125:130;pause(0.5); colormap gray;
    subplot(221);imagesc(squeeze(fliplr(flipud(abs(y1(:,i,:))))));
    subplot(222);imagesc(squeeze(fliplr(flipud(abs(y2(:,i,:))))));
    %subplot(223);imagesc(squeeze(fliplr(flipud(abs(y3(:,i,:))))));
    subplot(224);imagesc(squeeze(fliplr(flipud(abs(y(:,i,:))))));
end



for i=1:400;pause(0.1); colormap gray;
    subplot(131);imagesc(fliplr(flipud(abs(lowResRecons1(:,:,i)))));
    subplot(132);imagesc(fliplr(flipud(abs(lowResRecons2(:,:,i)))));
    %subplot(143);imagesc(fliplr(flipud(abs(lowResRecons3(:,:,i)))));
    subplot(133);imagesc(fliplr(flipud(abs(lowResRecons(:,:,i)))));

end
for i=1:400;pause(0.1); colormap gray;
    subplot(141);imagesc(fliplr(flipud(abs(x1(:,:,i)))));
     subplot(142);imagesc(fliplr(flipud(abs(x2(:,:,i)))));
%       subplot(143);imagesc(fliplr(flipud(abs(x3(:,:,i)))));
%       subplot(144);imagesc(fliplr(flipud(abs(x4(:,:,i)))));

end


for i=1:30;colormap gray;
    subplot(6,5,i);plot((((V(:,i)))));
end