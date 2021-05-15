addpath('./csm');
addpath('./utils');
addpath('./Utils');
addpath('./extras');
addpath('./nufft_toolbox_cpu');
addpath(genpath('./gpuNUFFT'));
addpath(genpath('./CUDA'));
load ('tempVR.mat');
FT_LR= NUFFT(ktraj_com,1,0,0,[N1,N1]);

lam=[1e-1 1e-2 1e-3 1e-4 1e-5];
for i=1:size(lam,2)

    i
    [~, tmp, ~] = est_laplacian_LR3(lowResRecons,FT_LR,kdata_com,csm_lowRes, N1,sigma, lam(i));

    save(strcat('VRlam_',num2str(lam(i)),'.mat'),'tmp');
end
