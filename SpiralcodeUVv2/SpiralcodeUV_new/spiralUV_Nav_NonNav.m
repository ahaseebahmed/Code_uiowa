 clear all

addpath(genpath('./../../Data/SpiralData_Virg/Phantom'));
addpath('./csm');
addpath('./utils');
addpath('./Utils');
addpath('./extras');
addpath('./nufft_toolbox_cpu');
addpath(genpath('./gpuNUFFT'));
addpath(genpath('./CUDA'));

%% Reconstruction parameters
spiralsToDelete=60;
framesToDelete=0;
ninterleavesPerFrame=6;
N = 300;
nChannelsToChoose=8;
numFramesToKeep = 200;
useGPU = 'true';
navigator='true';
SHRINK_FACTOR = 1.0;
nBasis = 30;
lambdaSmoothness = 0.025;
%cRatioI=[1:4,6:14,16:18,21:28,30:34];
%cRatioI=[1,2,3,8,9,11,12,13,14,15,16,17,19,20,21,22,23,24,25,26,29,30];
cRatioI=1:nChannelsToChoose;
sigma=[4.5];
lam=[0.1];
%%
% % ==============================================================
% % Load the data
% % ============================================================== 
 %load(strcat('Series27.mat'),'kdata','k','dcf');
 %dr1='./../../Data/SpiralData_2Oct18_UIOWA/Series8/';
 dr1='./../../Data/SpiralData_24Aug18_UIOWA/';

 dr2=pwd;
 cd(dr1);
 load('Series17.mat');
 cd(dr2);
 

%%
%kdata=kdata(:,:,:,1);
kdata = permute(kdata,[1,3,2]);


%------------Uni of virg----------%
% kdata=kdataUse;
%  k=ktrajmUse;
%  dcf=dcfUse;
%  clear kdataUse ktrajUse dcfUse;
 %%
 
 %-------------Preprocessing Data-------------%
 
[nFreqEncoding,nCh,numberSpirals]=size(kdata);
numFrames=floor((numberSpirals-spiralsToDelete)/ninterleavesPerFrame);
kdata=kdata(:,cRatioI(1:nChannelsToChoose),spiralsToDelete+1:numberSpirals);
k = k(:,spiralsToDelete+1:numberSpirals);
w=dcf(:,spiralsToDelete+1:numberSpirals);
kdata=kdata(:,:,1:numFrames*ninterleavesPerFrame);
k=k(:,1:numFrames*ninterleavesPerFrame);
w=w(:,1:numFrames*ninterleavesPerFrame);

kdata = permute(kdata,[1,3,2]);
kdata = reshape(kdata,[nFreqEncoding,ninterleavesPerFrame,numFrames,nChannelsToChoose]);
ktraj=k;
ktraj = reshape(ktraj,[nFreqEncoding,ninterleavesPerFrame,numFrames]);
w = reshape(w,[nFreqEncoding,ninterleavesPerFrame,numFrames]);

% Keeping only numFramesToKeep

kdata = kdata(:,:,framesToDelete+1:numFramesToKeep+framesToDelete,cRatioI(1:nChannelsToChoose));
ktraj = ktraj(:,:,framesToDelete+1:numFramesToKeep+framesToDelete);
%save data kdata ktraj dcf
%%

ktraj_scaled =  SHRINK_FACTOR*ktraj*N;
%% ==============================================================
% Compute the coil sensitivity map
% ============================================================== 
ktraj_scaled=reshape(ktraj_scaled,[nFreqEncoding,ninterleavesPerFrame,numFramesToKeep]);
kdata=reshape(kdata,[nFreqEncoding,ninterleavesPerFrame,numFramesToKeep,nChannelsToChoose]);
[coilImages] = coil_sens_map_NUFFT(kdata,ktraj_scaled,N,useGPU);

%% ===============================================================
% Compute coil compresession
% ================================================================
kdata=reshape(kdata,[nFreqEncoding*ninterleavesPerFrame*numFramesToKeep,nChannelsToChoose]);

[vkdata,vcoilImages] = combine_coilsv1(kdata,coilImages,0.9);
nChannelsToChoose=size(vcoilImages,3);
kdata=reshape(vkdata,[nFreqEncoding,ninterleavesPerFrame,numFramesToKeep,nChannelsToChoose]);
csm=giveEspiritMaps(reshape(vcoilImages,[size(vcoilImages,1), size(vcoilImages,2), nChannelsToChoose]),0.01);
coilImages=vcoilImages;

ktraj_scaled=reshape(ktraj_scaled,[nFreqEncoding,ninterleavesPerFrame,numFramesToKeep]);
kdata=reshape(kdata,[nFreqEncoding,ninterleavesPerFrame,numFramesToKeep,nChannelsToChoose]);
%% =========================================================
% % Iterative Strategy for non Navigator Data
% %=========================================================
ktraj_com=ktraj_scaled(:,2:end,:);
kdata_com=kdata(:,2:end,:,:);

%%if navigator==0
[kdata_com,ktraj_com] = binSpirals(kdata_com,ktraj_com,600,11,1);
N1 = 64;%ceil(max([max(abs(real(ktraj_com(:)))),max(abs(imag(ktraj_com(:))))]));
csm_lowRes = giveEspiritMapsSmall(coilImages,N1,N1);
ktraj_com = reshape(ktraj_com/N1,[size(ktraj_com,1),size(ktraj_com,2),numFramesToKeep]);
kdata_com = reshape(kdata_com, [size(kdata_com,1),size(ktraj_com,2),numFramesToKeep,nChannelsToChoose]);
FT_LR= NUFFT(ktraj_com,1,0,0,[N1,N1]);
tic;lowResRecons = l2Recont_v2(kdata_com,FT_LR,csm_lowRes,1e-9,N1);toc
lowResRecons=reshape(lowResRecons,[N1,N1,numFramesToKeep]);
% for ij=1:1
% [~, tmp, L] = est_laplacian_LR3(lowResRecons,FT_LR,kdata_com,csm_lowRes, N1,sigma, 1e-9);
% lowResRecons=tmp;
% end
[~,~,L]=estimateLapKernelLR(reshape(lowResRecons,[N1*N1,numFramesToKeep]),sigma,lam);

% lowResRecons=reshape(lowResRecons,[N1,N1,numFramesToKeep]);
% tmp=reshape(tmp,[N1,N1,numFramesToKeep]);
% tmp1=[lowResRecons,tmp];
% for i=1:530;imagesc(fliplr((abs(tmp1(:,:,i))))); pause(0.1); colormap gray;end
%% ==============================================================
% % Compute the weight matrix
% % ============================================================= 
%else
% no_ch=size(csm,3);
% Nav=permute((kdata(:,1,:,:)),[1,2,4,3]);
%Nav=Nav(:,:,1:264);
% Nav=reshape(permute(Nav,[1,3,2]),[nFreqEncoding*2,floor(numFramesToKeep/2),nChannelsToChoose]);
% Nav=permute(Nav,[1,3,2]);
%for ii=1:size(sigma,2)
 %   for jj=1:size(lam,2)
%[~,~,L]=estimateLapKernelLR(reshape(Nav,[nFreqEncoding*no_ch,numFramesToKeep]),sigma(ii),lam(jj));
[~,Sbasis,V]=svd(L);
V=V(:,end-nBasis+1:end);
Sbasis=Sbasis(end-nBasis+1:end,end-nBasis+1:end);
%% ==============================================================
% % Final Reconstruction
% % ============================================================= 
factor=0;
ktraj_scaled=reshape(ktraj_scaled,[nFreqEncoding*ninterleavesPerFrame,numFramesToKeep]);
kdata=reshape(kdata,[nFreqEncoding*ninterleavesPerFrame,numFramesToKeep,nChannelsToChoose]);
tic; x = solveUV(ktraj_scaled,kdata,csm, V, N,30,lambdaSmoothness*Sbasis,useGPU,factor);toc
% for i=1:1
%     factor=(abs(x)+mean(abs(x(:))).*10);
%     tic; x = solveUV(ktraj_scaled,kdata,csm, V, N, 30,lambdaSmoothness*Sbasis,useGPU,factor);toc
% end

y = reshape(reshape(x,[N*N,nBasis])*V',[N,N,numFramesToKeep]);

%% ==============================================================
% % Save and Display results
% % ============================================================= 

%for i=1:530;imagesc(fliplr(flipud(abs(lowResRecons(:,:,i))))); pause(0.1); colormap gray;end
%for i=1:250;imagesc(((abs(y(:,:,i))))); pause(0.1); colormap gray; end
%clear kdata csm V ;
%mkdir './../../Data/SpiralData_24Aug18_UIOWA/Series17/';
%cd './../../Data/SpiralData_Virg/Phantom/3T_';
cd(dr1);
save(strcat('ressf_iter.mat'), 'y','-v7.3');
cd(dr2);
%cd './../../../SpiralcodeUVv2/SpiralcodeUV_new';
   % end
%end

%end




% 
% for i=1:8;colormap gray;
%     subplot(2,4,i);imagesc(((abs(csm(:,:,i)))));
% end
% % figure;hold on;
%  for i=1:12
%      subplot(6,5,i);
%      plot(V(:,end+1-i));
%  end


% 
% tmp1 = reshape(tmp,[N1,N1,500]);
% load('lam_9e-09mat.mat')
% tmp2 = reshape(tmp,[N1,N1,500]);
% load('lam_5e-09mat.mat')
% tmp3 = reshape(tmp,[N1,N1,500]);
% load('lam_1e-09mat.mat')
% tmp4 = reshape(tmp,[N1,N1,500]);
% load('temp.mat','lowResRecons');
% tmp0 = reshape(lowResRecons,[N1,N1,500]);
% tt=[tmp1, tmp2];
% for i=1:500;imagesc(((abs(tt(:,:,i))))); pause(0.1); colormap gray;end