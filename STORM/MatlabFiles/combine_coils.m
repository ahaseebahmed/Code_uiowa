function [vKSpaceData,vCoilImages] = combine_coils(kSpaceData,coilimages,param,threshold)

    temp = permute(kSpaceData,[1,3,2]);
    temp = reshape(temp,param.n*param.nf*param.lines_per_SG_block,param.NcoilSelect);
    Rs = real(temp'*temp);


    [v,s] = eig(Rs);
    s = diag(s);[s,i] = sort(s,'descend');
    v = v(:,i);
    s=s./sum(s);s = cumsum(s);
    nvchannels = min(find(s>threshold));

    vKSpaceData = temp*v(:,1:nvchannels);
    
    vCoilImages = reshape(coilimages,param.n^2,param.NcoilSelect)*v(:,1:nvchannels);
    vCoilImages = reshape(vCoilImages, param.n,param.n,nvchannels);
    vKSpaceData = reshape(vKSpaceData,[param.n,param.lines_per_SG_block*param.nf,nvchannels]);
    vKSpaceData = permute(vKSpaceData,[1,3,2]);