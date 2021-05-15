function [vKSpaceData,vCoilImages] = combine_coilsv1(kSpaceData,coilImages,threshold)

    %temp = permute(kSpaceData,[1,3,2]);
    %temp = reshape(temp,param.n*param.nf*param.lines_per_SG_block,param.NcoilSelect);
   % coilImages=(coilImages./max(abs(coilImages(:))))*255;
    
    [N,~,NcoilSelect]=size(coilImages);
%      coilimages_roi=coilImages((N/2)-15:(N/2)+16,(N/2)-15:(N/2)+16,:);
%      coilimages_roi=reshape(coilimages_roi,[size(coilimages_roi,1)*size(coilimages_roi,2) NcoilSelect]);
%       coilImages=reshape(coilImages,[N*N NcoilSelect]);
%      me_an= mean(abs(coilimages_roi),1);
%      indx1=me_an>=(max(me_an(:))*0.15);
%      coilimages_roi=coilimages_roi(:,indx1(1:end));
%      coilImages=coilImages(:,indx1(1:end));
%      kSpaceData=kSpaceData(:,indx1);
     
%     for i=1:size(coilImages,2)
%         E(i)=(norm(squeeze(coilimages_roi(:,i))).^2)./norm(squeeze(coilImages(:,i))).^2;
%     end
%     indx2=E>=(max(E(:))*0.55);
%     coilImages=coilImages(:,indx2(1:end));
%     NcoilSelect=size(coilImages,2);
%     kSpaceData=kSpaceData(:,indx2);
    %coilImages=reshape(coilImages,[N,N,NcoilSelect]);
    
    %[N1,~,~]=size(coilimages);
    temp=kSpaceData;
    temp1=reshape(coilImages,[N^2,NcoilSelect]);
    Rs=real(temp1'*temp1);
    %Rs = real(temp'*temp);


    [v,s] = eig(Rs);
    s = diag(s);[s,i] = sort(s,'descend');
    v = v(:,i);
    s=s./sum(s);s = cumsum(s);
    nvchannels = min(find(s>threshold));

    vKSpaceData = temp*v(:,1:nvchannels);
    
    vCoilImages = reshape(coilImages,N^2,NcoilSelect)*v(:,1:nvchannels);
    vCoilImages = reshape(vCoilImages,N,N,nvchannels);
%     vKSpaceData = reshape(vKSpaceData,[param.n,param.lines_per_SG_block*param.nf,nvchannels]);
%     vKSpaceData = permute(vKSpaceData,[1,3,2]);