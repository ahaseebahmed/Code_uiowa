
function ssim=SSIM(imgR,true_im)

img1=gray2ind(mat2gray(abs(imgR)),256);
img2=gray2ind(mat2gray(abs(true_im)),256);

ssim=0;
for nn=1:size(imgR,3)
ssim= ssim+ssim_index(img1(:,:,nn), img2(:,:,nn));% 2D image compare

end

ssim=ssim/size(imgR,3);

end
 