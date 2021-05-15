function mask = giveMask(vCoilImages,sz,pk)

 q = sqrt(sum(abs(vCoilImages).^2,4));
 %p = abs(fftshift(q));
 %p = convn(q,ones(sz)/(sz^2),'same');
 if pk==0
    p = convn(q,ones(sz)/(sz^2),'same');
    p = p./max(p(:));
    pthresh = imfill(p >0.08,'holes');
    pthresh = double(imerode(pthresh,strel('disk',2)));
    mask = 1-convn(pthresh,ones(sz)/(sz^2),'same');
 else
    p=q;
    p = p./max(p(:));
 %pthresh = imfill(p >0.08,'holes');
    pthresh = double(imerode(p,strel('disk',sz)));
    mask = 1-pthresh;
 end

 
 
end