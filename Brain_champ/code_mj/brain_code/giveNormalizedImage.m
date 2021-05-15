function [normalized,biasfield] = giveNormalizedImage(input,threshold,lambda,useGPU)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

if(nargin <4)
    useGPU = true;
end
mask = abs(input)>threshold*max(input(:));
szMask = size(mask);
rangex = [-szMask(1)/2:szMask(1)/2-1];
rangey = [-szMask(2)/2:szMask(2)/2-1];
rangez = [-szMask(3)/2:szMask(3)/2-1];

[x,y,z] = meshgrid(rangex,rangey,rangez);
bigMask = ((x.^2+y.^2+z.^2)<szMask(1)^2/4);
if(useGPU)
    out = gpuArray(double(bigMask(:)));
else
    out = double(bigMask(:));
end
for i=1:4, 
    if(useGPU)
        currImage = gpuArray(double(input.*mask));
        ATA = @(x) reshape(currImage.*reshape(x,szMask),prod(szMask),1);
        R = @(x) laplacianMask(x,gpuArray(double(bigMask)));
    else
        currImage = (double(input.*mask));
        ATA = @(x) reshape(currImage.*reshape(x,szMask),prod(szMask),1);
        R = @(x) laplacianMask(x,(double(bigMask)));
    end
    
    Op = @(x) ATA(x) + lambda*R(x);
    [out,flag] = pcg(Op,gpuArray(double(mask(:))),1e-4,10);
    out = reshape(out,szMask).*bigMask;
    
    test = input.*out;
    mask = test > threshold*max(test(:));

    figure(1);imagesc(squeeze(abs(mask(:,171,:))));
    figure(2);imagesc(squeeze(abs(out(:,171,:))));
    figure(3);imagesc(squeeze(abs(test(:,171,:)))); colormap(gray);
    %figure(1);imagesc(abs(mask(:,:,floor(szMask(1)/2))))
    %figure(2);imagesc(abs(out(:,:,floor(szMask(1)/2))))
    %figure(3);imagesc(abs(test(:,:,floor(szMask(1)/2)))); colormap(gray);
 
    drawnow;
    if(flag==0)
        break;
    end
end

normalized = gather(test);
biasfield = gather(out);

end

