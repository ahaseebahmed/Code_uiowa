N=400;
nBasis=30;
numFramesToKeep=380;
for i=1:9
    load(strcat('res_26_',num2str(i),'.mat'));
 y = reshape(reshape(x,[N*N,nBasis])*V',[N,N,380]);
 y=rot90(flipud(abs(y)),1);
 
 res(:,:,:,i)=y(:,:,1:numFramesToKeep);
end

res=res(41:270,111:260,:,:);
save('Spiral_cine_3T_1leave_FB_026_full.mat','res');
