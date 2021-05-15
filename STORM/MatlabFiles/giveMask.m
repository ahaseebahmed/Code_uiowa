function mask = giveMask(vCoilImages,sz)

 q = sqrt(sum(abs(vCoilImages).^2,3));
 p = abs(fftshift(q));
 p = conv2(p,ones(sz)/(sz^2),'same');
 p = p./max(p(:));
 pthresh = imfill(p >0.08,'holes');
 pthresh = double(imerode(pthresh,strel('disk',sz)));
 mask = fftshift(1-conv2(pthresh,ones(sz)/(sz^2),'same'));
 
end