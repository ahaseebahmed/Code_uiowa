clear all

addpath('./../../Data');
addpath('./csm');
addpath('./nufft_toolbox_cpu');
addpath(genpath('./../../../Downloads/gpuNUFFT-master/gpuNUFFT'));
addpath(genpath('./../../../Downloads/gpuNUFFT-master/CUDA/bin'));


spiralsToDelete=60;
ninterleavesPerFrame=6;
N = 300;
nChannelsToChoose=4;
nBasis=30;
Nonfft = true;
numFramesToKeep = 490;
%% Reconstruction parameters
maxitCG = 15;
alpha = 1e-5;
tol = 1e-6;
display = 0;
imwidth = N;
useGPU = 'true';
SHRINK_FACTOR = 1.3;
nBasis = 30;
%%
% % %
% % ==============================================================
% % Load the data
% % ============================================================== 
%Load data
 load('./../../Data/Spiral_cine_3T_1leave_FB_027_full.mat','kdata','ktraj','dcf');
 load('./../../Data/cRatioI_027.mat');
 kdata=squeeze(kdata);
 
 
 [nFreqEncoding,nCh,numberSpirals]=size(kdata);
 numFrames=floor((numberSpirals-spiralsToDelete)/ninterleavesPerFrame);
 

 kdata=kdata(:,cRatioI(1:nChannelsToChoose),spiralsToDelete+1:numberSpirals);
 ktraj=cell2mat(ktraj);
 ktraj = ktraj(:,spiralsToDelete+1:numberSpirals);
 dcf=ones(nFreqEncoding,1);
% Reshaping to frames

kdata = permute(kdata,[1,3,2]);
kdata = reshape(kdata,[nFreqEncoding,ninterleavesPerFrame,numFrames,nChannelsToChoose]);
ktraj = reshape(ktraj,[nFreqEncoding,ninterleavesPerFrame,numFrames]);
% Keeping only numFramesToKeep

kdata = kdata(:,:,1:numFramesToKeep,:);
ktraj = ktraj(:,:,1:numFramesToKeep);
%save data kdata ktraj dcf
%%

[nFreqEncoding,numberSpirals,~,~]=size(kdata);
ktraj_scaled =  SHRINK_FACTOR*ktraj;
%% ==============================================================
% Compute the coil sensitivity map
% ============================================================== 
ktraj_scaled=reshape(ktraj_scaled,[nFreqEncoding,ninterleavesPerFrame,numFramesToKeep]);
kdata=reshape(kdata,[nFreqEncoding,ninterleavesPerFrame,numFramesToKeep,nChannelsToChoose]);
[csm,coilImages] = coil_sens_map_NUFFT(kdata,ktraj_scaled,N,dcf,useGPU);

% % ==============================================================
% % Compute the weight matrix
% % ============================================================== 
%%
ktraj_scaled=reshape(ktraj_scaled,[nFreqEncoding,ninterleavesPerFrame,numFramesToKeep]);
kdata=reshape(kdata,[nFreqEncoding,ninterleavesPerFrame,numFramesToKeep,nChannelsToChoose]);

[kdata_com,ktraj_com] = binSpirals(kdata,ktraj_scaled,1200,7);

tic;lowResRecons = l2Recon(kdata_com,ktraj_com,coilImages,0.01,dcf);toc
[N1,~,~] = size(lowResRecons);
[~,~,L]=estimateLapKernelLR(reshape(lowResRecons,[N1*N1,numFramesToKeep]),1e-5,1);
[~,Sbasis,V]=svd(L);
V=V(:,end-nBasis+1:end);
Sbasis=Sbasis(end-nBasis+1:end,end-nBasis+1:end);
lambdaSmoothness = 0.02;

clear kdata_com ktraj_com

ktraj_scaled=reshape(ktraj_scaled,[nFreqEncoding*ninterleavesPerFrame,numFramesToKeep]);
kdata=reshape(kdata,[nFreqEncoding*ninterleavesPerFrame,numFramesToKeep,nChannelsToChoose]);
tic; x = solveUV(ktraj_scaled,kdata,csm, V, N, 20,lambdaSmoothness*Sbasis,dcf,useGPU);toc
y = reshape(reshape(x,[N*N,nBasis])*V',[N,N,numFramesToKeep]);

