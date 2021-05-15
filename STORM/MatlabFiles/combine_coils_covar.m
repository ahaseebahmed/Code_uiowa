function [vKSpaceData,vCoilImages] = combine_coils_covar(kSpaceData,coilimages1,coilimages2,param,threshold)

    signal = reshape(coilimages1+coilimages2,param.n*param.n,param.nchannels)/2;
    noise = reshape(coilimages1-coilimages2,param.n*param.n,param.nchannels);

    Rn = noise'*noise;
    Rs = signal'*signal;
    P = pinv(Rn)*Rs;
    [u,s,v] = svd(P);
    s = diag(s);

    temp = permute(kSpaceData,[1,3,2]);
    temp = reshape(temp,param.n*param.nf*param.lines_per_SG_block,param.nchannels);
    s=s./sum(s);s = cumsum(s);
    nvchannels = min(find(s>threshold));

    vKSpaceData = temp*v(:,1:nvchannels);
    
    coilimages = (coilimages1 + coilimages2)/2;
    vCoilImages = reshape(coilimages,param.n^2,param.NcoilSelect)*v(:,1:nvchannels);
    vCoilImages = reshape(vCoilImages, param.n,param.n,nvchannels);
    vKSpaceData = reshape(vKSpaceData,[param.n,param.lines_per_SG_block*param.nf,nvchannels]);
    vKSpaceData = permute(vKSpaceData,[1,3,2]);