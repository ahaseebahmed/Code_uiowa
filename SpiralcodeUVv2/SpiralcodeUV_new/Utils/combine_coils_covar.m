function [vKSpaceData,vCoilImages] = combine_coils_covar(kSpaceData,coilimages1,coilimages2,threshold)

   [N,~,NcoilSelect]=size(coilimages1);

    signal = reshape(coilimages1+coilimages2,N*N,NcoilSelect)/2;
    noise = reshape(coilimages1-coilimages2,N*N,NcoilSelect);

    Rn = noise'*noise;
    Rs = signal'*signal;
    P = pinv(Rn)*Rs;
    [u,s,v] = svd(P);
    s = diag(s);

    temp=kSpaceData;
    %temp1=reshape(coilimages,[N^2,NcoilSelect]);
    s=s./sum(s);s = cumsum(s);
    nvchannels = min(find(s>threshold));

    vKSpaceData = temp*v(:,1:nvchannels);
    
    coilimages = (coilimages1 + coilimages2)/2;
    vCoilImages = reshape(coilimages,N^2,NcoilSelect)*v(:,1:nvchannels);
    vCoilImages = reshape(vCoilImages,N,N,nvchannels);
 