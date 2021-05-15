addpath('./csm');
addpath('./utils');
addpath('./Utils');
addpath('./extras');
addpath('./nufft_toolbox_cpu');
addpath(genpath('./gpuNUFFT'));
addpath(genpath('./CUDA'));
load ('temp.mat');
FT_LR= NUFFT(ktraj_com,1,0,0,[N1,N1]);

lam=[0.6];
for ii=1:size(lam,2)

    ii
    [~, tmp, ~] = est_laplacian_LR3(lowResRecons,FT_LR,kdata_com,csm_lowRes, N1,0.88, lam(ii));
%[~,tmp,L]=estimateLapKernelLR(lowResRecons,0.5,0.3);
%[~,Sbasis,V]=svd(L);
    save(strcat('lam_',num2str(lam(ii)),'.mat'),'tmp');
end
