clear all

addpath(genpath('./../../Data/SpiralData_Virg/Phantom'));
addpath('./csm');
addpath('./utils');
addpath('./Utils');
addpath('./extras');
addpath('./nufft_toolbox_cpu');
addpath(genpath('./gpuNUFFT'));
addpath(genpath('./../../Wavelab850'));
%addpath(genpath('./CUDA'));

%% Reconstruction parameters
spiralsToDelete=200%60;
ninterleavesPerFrame=1;
N = 160;
nChannels=16;
nBasis=30;
Nonfft = true;
numFramesToKeep = 3000;
%chh=[5,7,9,10,17,21,22,29,30];
chh=1:34;
cRatioI=1:nChannels;
maxitCG = 15;
useGPU = true;
SHRINK_FACTOR = 1;
%nFreqEncoding=512;
%numberSpirals=10000;
nChannelsToChoose=16;
framesToDelete=0;
sigma=4.5;
lam=0.01;
lambdaSmoothness=0.001;
%%
% % ==============================================================
% % Load the data
% % ============================================================== 
 %load(strcat('Series27.mat'),'kdata','k','dcf');
 dr1='./../../Data/SpeechData';
 dr2=pwd;
 cd(dr1);
  kdata1=load('speech_example_vds.mat');

% load('Series6.mat');
 cd(dr2);
kdata=kdata1.kdata;
k=kdata1.k;
dcf=kdata1.dcf;
for sl=1:1%size(kdata1,4)
%%
kdata=kdata(:,:,:,sl);
kdata=kdata/max(abs(kdata(:)));
%kdata = permute(kdata,[1,3,2]);

%delta = 0;%0.2; % shift in k-space for eddy current adjustment
%k = giveTinyGoldenAngleTraj(nFreqEncoding,numberSpirals,delta);

%------------Uni of virg----------%
% kdata=kdataUse;
%  k=ktrajmUse;
%  dcf=dcfUse;
%  clear kdataUse ktrajUse dcfUse;
 %%
 
 %-------------Preprocessing Data-------------%
 kdata=permute(kdata,[2,1,3]);
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
%w = reshape(w,[nFreqEncoding,ninterleavesPerFrame,numFrames]);

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
% coilImages=coilImages(:,:,chh(1:size(chh,2)));
% kdata=kdata(:,:,:,chh(1:size(chh,2)));
% nChannelsToChoose=size(chh,2);
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
%if navigator==0
    
%[kdata_com,ktraj_com] = binSpirals(kdata,ktraj_scaled,128,5,1);
kdata_com=kdata(1:128,1,:,:);
ktraj_com=ktraj_scaled(1:128,1,:,:);

N1 = 60;%ceil(max([max(abs(real(ktraj_com(:)))),max(abs(imag(ktraj_com(:))))]));
csm_lowRes = giveEspiritMapsSmall(coilImages,N1,N1);
ktraj_com = reshape(ktraj_com/N1,[size(ktraj_com,1),size(ktraj_com,2),numFramesToKeep]);
kdata_com = reshape(kdata_com, [size(kdata_com,1),size(ktraj_com,2),numFramesToKeep,nChannelsToChoose]);
FT_LR= NUFFT(ktraj_com,1,0,0,[N1,N1]);
WT=Wavelet('Daubechies',4,4);
tic;lowResRecons = l2Recont_v2(kdata_com,FT_LR,csm_lowRes,0.1,N1);toc
lowResRecons=reshape(lowResRecons,[N1*N1,numFramesToKeep]);
[~,~,L]=estimateLapKernelLR(lowResRecons,sigma,lam);
[ tmp, L] = iterative_est_laplacian1(L,FT_LR,WT,kdata_com,csm_lowRes, N1,sigma, lam);
%% ==============================================================
% % Compute the weight matrix
% % ============================================================= 
%if navigator
% no_ch=size(csm,3);
% Nav=permute((kdata(:,1,:,:)),[1,2,4,3]);
% %Nav=Nav(:,:,1:264);
% % Nav=reshape(permute(Nav,[1,3,2]),[nFreqEncoding*2,floor(numFramesToKeep/2),nChannelsToChoose]);
% % Nav=permute(Nav,[1,3,2]);
% %for ii=1:size(sigma,2)
%  %   for jj=1:size(lam,2)
%tmp=reshape(tmp,60,60,1000);
%[~,~,L]=estimateLapKernelLR(tmp,sigma,lam);
% %end
[~,Sbasis,V]=svd(L);
V=V(:,end-nBasis+1:end);
Sbasis=Sbasis(end-nBasis+1:end,end-nBasis+1:end);
%% ==============================================================
% % Final Reconstruction
% % ============================================================= 
factor=0;
ktraj_scaled=reshape(ktraj_scaled,[nFreqEncoding*ninterleavesPerFrame,numFramesToKeep]);
kdata=reshape(kdata,[nFreqEncoding*ninterleavesPerFrame,numFramesToKeep,nChannelsToChoose]);
tic; x = solveUV(ktraj_scaled,kdata,csm, V, N, 70,lambdaSmoothness*Sbasis,useGPU,factor);toc
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
save(strcat('datan_','se6v2_',num2str(sl),'.mat'), 'y','-v7.3');
cd(dr2);
%cd './../../../SpiralcodeUVv2/SpiralcodeUV_new';
   % end
%end

end




% 
% for i=1:16;colormap jet;
%     subplot(4,4,i);imagesc(((abs(coilImages(:,:,i)))));
% end
% % figure;hold on;
%  for i=1:12
%      subplot(6,5,i);
%      plot(V(:,end+1-i));
%  end