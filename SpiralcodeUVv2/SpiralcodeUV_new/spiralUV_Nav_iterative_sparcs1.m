clear all

%addpath(genpath('./../../'));
addpath('./csm');
addpath('./utils');
addpath('./Utils');
addpath('./extras');
addpath('./nufft_toolbox_cpu');
addpath(genpath('./gpuNUFFT'));
addpath(genpath('./../../Wavelab850'));
%addpath(genpath('./CUDA'));

%% Reconstruction parameters
spiralsToDelete=100%60;
ninterleavesPerFrame=5;
N = 400;
nBasis=30;
numFramesToKeep =1580;
chh=1:34;
maxitCG = 15;
useGPU = true;
SHRINK_FACTOR = 1;
nChannelsToChoose=30;
framesToDelete=0;
sigma=4.5;
lam=0.1;
lambdaSmoothness=0.009;
%%
% % ==============================================================
% % Load the data
% % ============================================================== 
 %load(strcat('Series27.mat'),'kdata','k','dcf');
 dr1='./../../../Sparcs';
 dr2=pwd;
 for sl=6:6
 cd(dr1);
 load(strcat('Spiral_cine_3T_1leave_FB_015_slice',num2str(sl),'_full.mat'),'kdata','ktraj','dcf');
 cd(dr2);

%%
kdata=squeeze(kdata);
kdata=kdata/max(abs(kdata(:)));
%kdata = permute(kdata,[1,3,2]);
k=cell2mat(ktraj);
 %% %-------------Preprocessing Data-------------%
 
[nFreqEncoding,nCh,numberSpirals]=size(kdata);
numFrames=floor((numberSpirals-spiralsToDelete)/ninterleavesPerFrame);
kdata=kdata(:,:,spiralsToDelete+1:numberSpirals);
k = k(:,spiralsToDelete+1:numberSpirals);
%w=dcf(:,spiralsToDelete+1:numberSpirals);
kdata=kdata(:,:,1:numFrames*ninterleavesPerFrame);
k=k(:,1:numFrames*ninterleavesPerFrame);
%w=w(:,1:numFrames*ninterleavesPerFrame);
kdata = permute(kdata,[1,3,2]);
nChannelsToChoose=nCh;
kdata = reshape(kdata,[nFreqEncoding,ninterleavesPerFrame,numFrames,nChannelsToChoose]);
ktraj=k;
ktraj = reshape(ktraj,[nFreqEncoding,ninterleavesPerFrame,numFrames]);
%w = reshape(w,[nFreqEncoding,ninterleavesPerFrame,numFrames]);

% Keeping only numFramesToKeep

kdata = kdata(:,:,1:numFramesToKeep,:);
ktraj = ktraj(:,:,1:numFramesToKeep);
%save data kdata ktraj dcf
%%
ktraj_scaled =  SHRINK_FACTOR*ktraj/(2*max(abs(ktraj(:))));
%ktraj_scaled =  0.5*ktraj;%*N;
%% ==============================================================
% Compute the coil sensitivity map
% ============================================================== 
ktraj_scaled=reshape(ktraj_scaled,[nFreqEncoding,ninterleavesPerFrame,numFramesToKeep]);
kdata=reshape(kdata,[nFreqEncoding,ninterleavesPerFrame,numFramesToKeep,nChannelsToChoose]);
[coilImages1] = coil_sens_map_NUFFT(kdata(:,:,1:numFramesToKeep/2,:),ktraj_scaled(:,:,1:numFramesToKeep/2),N,useGPU);
[coilImages2] = coil_sens_map_NUFFT(kdata(:,:,1+(numFramesToKeep/2):end,:),ktraj_scaled(:,:,1+(numFramesToKeep/2):end),N,useGPU);

%% ===============================================================
% Compute coil compresession
% ================================================================
kdata=reshape(kdata,[nFreqEncoding*ninterleavesPerFrame*numFramesToKeep,nChannelsToChoose]);
[vkdata,vcoilImages] = combine_coils_covar(kdata,coilImages1,coilImages2,0.85);

%[vkdata,vcoilImages] = combine_coilsv1(kdata,coilImages,0.85);
%[vKSpaceData,vCoilImages] = combine_coils_covar(kSpaceData,coilimages1,coilimages2,param,threshold)
nChannelsToChoose=size(vcoilImages,3);
kdata=reshape(vkdata,[nFreqEncoding,ninterleavesPerFrame,numFramesToKeep,nChannelsToChoose]);
csm=giveEspiritMaps(reshape(vcoilImages,[size(vcoilImages,1), size(vcoilImages,2), nChannelsToChoose]),0.005);
coilImages=vcoilImages;
ktraj_scaled=reshape(ktraj_scaled,[nFreqEncoding,ninterleavesPerFrame,numFramesToKeep]);
kdata=reshape(kdata,[nFreqEncoding,ninterleavesPerFrame,numFramesToKeep,nChannelsToChoose]);
%% =========================================================
% % Iterative Strategy for non Navigator Data
% %=========================================================
%if navigator==0
    
% [kdata_com,ktraj_com] = binSpirals(kdata,ktraj_scaled,600,5,1);
% ktraj_com=ktraj_com/(2*max(abs(ktraj_com(:))));
% N1 = 64;%ceil(max([max(abs(real(ktraj_com(:)))),max(abs(imag(ktraj_com(:))))]));
% csm_lowRes = giveEspiritMapsSmall(coilImages,N1,N1);
% ktraj_com = reshape(ktraj_com,[size(ktraj_com,1),size(ktraj_com,2),numFramesToKeep]);
% kdata_com = reshape(kdata_com, [size(kdata_com,1),size(ktraj_com,2),numFramesToKeep,nChannelsToChoose]);
% FT_LR= NUFFT(ktraj_com,1,0,0,[N1,N1]);
% WT=Wavelet('Daubechies',4,4);
% tic;lowResRecons = l2Recont_v2(kdata_com,FT_LR,csm_lowRes,0.01,N1);toc
% lowResRecons=reshape(lowResRecons,[N1*N1,numFramesToKeep]);
% %  q= fft(lowResRecons,[],2);
% % q(:,numFramesToKeep/2-numFramesToKeep/4-150:numFramesToKeep/2+numFramesToKeep/4+150)=0;
% % im_nav_smoothed = ifft(q,[],2);
% %[ ~,X,~]=est_laplacian_rev(lowResRecons,FT_LR,kdata_com,csm_lowRes,N1, sigma, lam);
% %[~, tmp, L] = est_laplacian_LR3(lowResRecons,FT_LR,kdata_com,csm_lowRes,N1, sigma, lam);
% [~,~,L]=estimateLapKernelLR(lowResRecons,sigma,lam);
% [ X, ~] = iterative_est_laplacian1(L,FT_LR,WT,kdata_com,csm_lowRes, N1,sigma, lam);
%   q= fft(X,[],2);
%  q(:,numFramesToKeep/2-numFramesToKeep/4-55:numFramesToKeep/2+numFramesToKeep/4+55)=0;
%  X1 = ifft(q,[],2);
cd(dr1);
load(strcat('res_15_',num2str(sl),'.mat'));
cd(dr2);
X1 = reshape(x,[N*N,nBasis])*V';
X1=X1(:,1:numFramesToKeep);


[~,~,L]=estimateLapKernelLR(X1,4.5,0.001);
%% ==============================================================
% % Compute the weight matrix
% % ============================================================= 
[~,Sbasis,V]=svd(L);
V=V(:,end-nBasis+1:end);
Sbasis=Sbasis(end-nBasis+1:end,end-nBasis+1:end);
%% ==============================================================
% % Final Reconstruction
% % ============================================================= 
factor=0;
ktraj_scaled=reshape(ktraj_scaled,[nFreqEncoding*ninterleavesPerFrame,numFramesToKeep]);
kdata=reshape(kdata,[nFreqEncoding*ninterleavesPerFrame,numFramesToKeep,nChannelsToChoose]);
tic; x = solveUV(ktraj_scaled,kdata,csm, V, N, 60,lambdaSmoothness*Sbasis,useGPU,factor);toc
% for i=1:1
%     factor=(abs(x)+mean(abs(x(:))).*10);
%     tic; x = solveUV(ktraj_scaled,kdata,csm, V, N, 30,lambdaSmoothness*Sbasis,useGPU,factor);toc
% end

y = reshape(reshape(x,[N*N,nBasis])*V',[N,N,numFramesToKeep]);

%% ==============================================================
% % Save and Display results
% % ============================================================= 

%for i=1:530;imagesc(fliplr(flipud(abs(y(:,:,i))))); pause(0.1); colormap gray;end
%for i=1:250;imagesc(((abs(y(:,:,i))))); pause(0.1); colormap gray; end
%clear kdata csm V ;
%mkdir './../../Data/SpiralData_24Aug18_UIOWA/Series17/';
%cd './../../Data/SpiralData_Virg/Phantom/3T_';
cd(dr1);
save(strcat('resn_14_',num2str(sl),'.mat'), 'x','V','-v7.3');
cd(dr2);
 end
%cd './../../../SpiralcodeUVv2/SpiralcodeUV_new';
   % end
%end




% 
% for i=1:3;colormap jet;
%     subplot(1,3,i);imagesc(((abs(vcoilImages(:,:,i)))));
% end
% % figure;hold on;
%  for i=1:12
%      subplot(6,5,i);
%      plot(V(:,end+1-i));
%  end