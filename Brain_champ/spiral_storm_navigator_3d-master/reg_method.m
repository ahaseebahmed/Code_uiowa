[optimizer, metric] = imregconfig('monomodal');
img=img/max(abs(img(:)));
img1=abs(img);
for i= 2: 16
    imreg(:,:,:,i)=imregister(img1(:,:,:,i),img1(:,:,:,1),'rigid',optimizer,metric);
end

    