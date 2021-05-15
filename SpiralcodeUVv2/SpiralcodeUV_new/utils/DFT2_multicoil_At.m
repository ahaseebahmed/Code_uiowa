function res=DFT2_multicoil_At(x,At_DFT,CSM,n,no_frames,no_channels,V)
% % Inputs:
% %     x:  k-space data with dimension is no of pts x no. interleaves 
% %         no. of frames x no. of channels  
% %     n is the image dimension along one direction
% %     csm: coil sensitivity map: dimension:n x n x no of coils
% %     no_channels : no. of channels
% %     no_frames : no. of frames
% %     V: temporal basis matrix: dimension: no of frames x no. of basis
% %     
% % Ouputs:
% %     res: U initial: dimension: N x N x no. of basis


nbasis=size(V,2);
res = zeros(n,n);
res1 = zeros(n*n,nbasis);

for i=1:no_frames
    for j=1:no_channels
        idata_t=double(At_DFT{i}(x(:,j,i)));
        res = res + conj(CSM(:,:,j)).*idata_t;
    end
    res1=res1+reshape(res,n^2,1)*V(i,:);
    res(:)=0;
end

res = n*res1(:);












