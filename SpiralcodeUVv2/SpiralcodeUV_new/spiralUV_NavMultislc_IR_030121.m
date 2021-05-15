 %clear all

addpath(genpath('./../../Data/SpiralData_Virg/Phantom'));
addpath('./csm');
addpath('./utils');
addpath('./Utils');
addpath('./extras');
addpath('./nufft_toolbox_cpu');
addpath(genpath('./gpuNUFFT'));
addpath(genpath('./CUDA'));

%% Reconstruction parameters
spiralsToDelete=0;
framesToDelete=0;
ninterleavesPerFrame=4;
N = 340;
nChannelsToChoose=14;
numFramesToKeep = 1000;
useGPU = 'true';
SHRINK_FACTOR = 1.0;
nBasis = 30;
lambdaSmoothness = 0.008;
cRatioI=[2,3,4,7,8,9,10,15,16,17,22,27,28,29];
%cRatioI=[1,2,3,8,9,11,12,13,14,15,16,17,19,20,21,22,23,24,25,26,29,30];
%cRatioI=1:nChannelsToChoose;
sigma=[4.5];
lam=[0.001];
% num_spiral_IR=700;
% num_IR=9;
%%
% % ==============================================================
% % Load the data
% % ============================================================== 
 %load(strcat('Series27.mat'),'kdata','k','dcf');
% dr1='./../../Data/SpiralData_2Oct18_UIOWA/Series10/';
 dr1=['/Shared/lss_jcb/DMRI_code/NUFFT/IRVirginia/'];

 dr2=pwd;
 cd(dr1);
 load('Spiral_cine_3T_1leave_FB_062_IR_FA15_TBW5_6_fixedSpiral_post_full.mat','kdata','ktraj','dcf');
 cd(dr2);
 
 kdata=squeeze(kdata);
 ktraj=cell2mat(ktraj);
 %%
 
 %-------------Preprocessing Data-------------%
 
[nFreqEncoding,nCh,numberSpirals]=size(kdata);
numFrames=floor((numberSpirals-spiralsToDelete)/ninterleavesPerFrame);
kdata=kdata(:,:,1:numFrames*ninterleavesPerFrame);
ktraj=ktraj(:,1:numFrames*ninterleavesPerFrame);
kdata = permute(kdata,[1,3,2]);
kdata = reshape(kdata,[nFreqEncoding,ninterleavesPerFrame,numFrames,nCh]);
ktraj = reshape(ktraj,[nFreqEncoding,ninterleavesPerFrame,numFrames]);
%w = reshape(w,[nFreqEncoding,ninterleavesPerFrame,numFrames]);

% Keeping only numFramesToKeep

kdata = kdata(:,:,framesToDelete+1:numFramesToKeep+framesToDelete,cRatioI(1:nChannelsToChoose));
ktraj = ktraj(:,:,framesToDelete+1:numFramesToKeep+framesToDelete);
%save data kdata ktraj dcf
%%

%ktraj_scaled =  SHRINK_FACTOR*ktraj*N;
ktraj_scaled =  SHRINK_FACTOR*ktraj/(2*max(abs(ktraj(:))));
%ktraj_scaled =  SHRINK_FACTOR*ktraj;


%% ==============================================================
% Compute the coil sensitivity map
% ============================================================== 
ktraj_scaled=reshape(ktraj_scaled,[nFreqEncoding,ninterleavesPerFrame,numFramesToKeep]);
kdata=reshape(kdata,[nFreqEncoding,ninterleavesPerFrame,numFramesToKeep,nChannelsToChoose]);
%[coilImages] = coil_sens_map_NUFFT(kdata,ktraj_scaled,N,useGPU);
[coilImages1] = coil_sens_map_NUFFT(kdata(:,:,1:numFramesToKeep/2,:),ktraj_scaled(:,:,1:numFramesToKeep/2),N,useGPU);
[coilImages2] = coil_sens_map_NUFFT(kdata(:,:,1+(numFramesToKeep/2):end,:),ktraj_scaled(:,:,1+(numFramesToKeep/2):end),N,useGPU);

%% ===============================================================
% Compute coil compresession
% ================================================================
kdata=reshape(kdata,[nFreqEncoding*ninterleavesPerFrame*numFramesToKeep,nChannelsToChoose]);
[vkdata,vcoilImages] = combine_coils_covar(kdata,coilImages1,coilImages2,0.85);
nChannelsToChoose=size(vcoilImages,3);
coilImages=vcoilImages;
kdata=reshape(vkdata,[nFreqEncoding,ninterleavesPerFrame,numFramesToKeep,nChannelsToChoose]);
csm=giveEspiritMaps(reshape(coilImages,[size(coilImages,1), size(coilImages,2), nChannelsToChoose]),0.0025);
%coilImages=vcoilImages;

ktraj_scaled=reshape(ktraj_scaled,[nFreqEncoding,ninterleavesPerFrame,numFramesToKeep]);
kdata=reshape(kdata,[nFreqEncoding,ninterleavesPerFrame,numFramesToKeep,nChannelsToChoose]);

%% ==============================================================
% % Compute the weight matrix
% % ============================================================= 
no_ch=nChannelsToChoose;
Nav=squeeze(permute((kdata(:,1,:,:)),[1,2,4,3]));
Nav=Nav/max(abs(Nav(:)));
Nav=reshape(Nav,[nFreqEncoding*no_ch,numFramesToKeep]);

 q= fft(Nav,[],2);
 q(:,numFramesToKeep/2-numFramesToKeep/4-100:numFramesToKeep/2+numFramesToKeep/4+100)=0;
 Nav = ifft(q,[],2);
%Nav=Nav(:,:,1:264);
% Nav=reshape(permute(Nav,[1,3,2]),[nFreqEncoding*2,floor(numFramesToKeep/2),nChannelsToChoose]);
% Nav=permute(Nav,[1,3,2]);
[K,X,L]=estimateLapKernelLR(reshape(Nav,[nFreqEncoding*no_ch,numFramesToKeep]),4.5,0.001);
%load('/Shared/lss_jcb/abdul/IRdata/L_IR.mat');
[~,Sbasis,V]=svd(L);
V=V(:,end-nBasis+1:end);
Sbasis=Sbasis(end-nBasis+1:end,end-nBasis+1:end);
%% ==============================================================
% % Final Reconstruction
% % ============================================================= 
ktraj_scaled=reshape(ktraj_scaled,[nFreqEncoding*ninterleavesPerFrame,numFramesToKeep]);
kdata=reshape(kdata,[nFreqEncoding*ninterleavesPerFrame,numFramesToKeep,nChannelsToChoose]);
dcf=repmat(dcf,[1,ninterleavesPerFrame,numFramesToKeep]);
dcf=reshape(dcf,[nFreqEncoding*ninterleavesPerFrame,numFramesToKeep]);

tic; x = solveUV(ktraj_scaled,dcf1,kdata,csm, V, N, 100,lambdaSmoothness*Sbasis,useGPU);toc
y = reshape(reshape(x,[N*N,nBasis])*V',[N,N,numFramesToKeep]);

%% ==============================================================
% % Save and Display results
% % ============================================================= 

%for i=1:530;imagesc(fliplr(flipud(abs(y(:,:,i))))); pause(0.1); colormap gray;end
%for i=1:250;imagesc(rot90(flipud(abs(y(:,:,i))))); pause(0.1); colormap gray; end
%clear kdata csm V ;
%mkdir './../../Data/SpiralData_24Aug18_UIOWA/Series17/';
%cd './../../Data/SpiralData_Virg/Phantom/3T_';
cd(dr1);
save(strcat('Spiral_cine_3T_1leave_FB_062_IR_FA15_TBW5_6_fixedSpiral_post_full28',num2str(lambdaSmoothness),'_',num2str(sigma),'_',num2str(lam),'.mat'), 'y','L','-v7.3');
cd(dr2);
%cd './../../../SpiralcodeUVv2/SpiralcodeUV_new';

%end

% 
% for i=1:8;colormap gray;
%     subplot(2,4,i);imagesc(((abs(csm(:,:,i)))));
% end
% figure;hold on;
%  for i=1:30
%      subplot(6,5,i);
%      plot(V(:,end+1-i));
%  end
%50:270,106:230 IR data