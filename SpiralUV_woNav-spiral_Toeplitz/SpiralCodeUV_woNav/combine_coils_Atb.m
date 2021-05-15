function [vAtb,vCoilImages] = combine_coils_Atb(Atb,coilimages,threshold,coilImagesRoi)

    [N,~,Nframes,Ncoils]=size(Atb);
    
    if(nargin>3)
        [Nx,Ny,~] = size(coilImagesRoi);
        temp=reshape(coilImagesRoi,Nx*Ny,Ncoils);
        Rs = real(temp'*temp);
        temp=reshape(Atb,N*N*Nframes,Ncoils);
    else
        temp=reshape(Atb,N*N*Nframes,Ncoils);
        Rs = real(temp'*temp);
    end

    [v,s] = eig(Rs);
    s = diag(s);[s,i] = sort(s,'descend');
    v = v(:,i);
    s=s./sum(s);s = cumsum(s);
    nvchannels = min(find(s>threshold));

    vAtb = reshape(temp*v(:,1:nvchannels),[N,N,Nframes,nvchannels]);
    
    vCoilImages = reshape(coilimages,N^2,Ncoils)*v(:,1:nvchannels);
    vCoilImages = reshape(vCoilImages,N,N,nvchannels);