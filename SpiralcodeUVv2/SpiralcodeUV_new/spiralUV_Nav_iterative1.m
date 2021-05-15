clear all
addpath(genpath('./../../Data/SpiralData_Virg/Phantom'));
addpath('./csm');
addpath('./utils');
addpath('./Utils');
addpath('./extras');
addpath('./nufft_toolbox_cpu');
addpath(genpath('./gpuNUFFT'));
addpath(genpath('./CUDA'));
addpath(genpath('./../../Wavelab850'));

%% Reconstruction parameters
spiralsToDelete=0;
framesToDelete=0;
ninterleavesPerFrame=16;
N = 340;
nChannelsToChoose=30;
numFramesToKeep = 250;
useGPU = 'true';
SHRINK_FACTOR = 1.0;
nBasis = 20;
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
 dr1='./../../Data/SpiralData_UVirginia/HumanIR';
 dr2=pwd;
 cd(dr1);
  load('Spiral_cine_3T_1leave_FB_062_IR_FA15_TBW5_6_fixedSpiral_post_full.mat','kdata','ktraj','dcf');
 %ksp=twix_obj{2}.image();
% load('Series6.mat');
 cd(dr2);
 kdata=squeeze(kdata);
  k=cell2mat(ktraj);
% for sl=1:1%size(kdata1,4)
% %%
% kdata=kdata1(:,:,:,sl);
kdata=kdata/max(abs(kdata(:)));
%kdata = permute(kdata,[1,3,2]);

%delta = 0; % shift in k-space for eddy current adjustment
%k = giveTinyGoldenAngleTraj(nFreqEncoding,numberSpirals,delta);

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
%w=dcf(:,spiralsToDelete+1:numberSpirals);
kdata=kdata(:,:,1:numFrames*ninterleavesPerFrame);
k=k(:,1:numFrames*ninterleavesPerFrame);
%w=w(:,1:numFrames*ninterleavesPerFrame);

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

%ktraj_scaled =  SHRINK_FACTOR*ktraj;%*N;
ktraj_scaled =  SHRINK_FACTOR*ktraj/(2*max(abs(ktraj(:))));

%% ==============================================================
% Compute the coil sensitivity map
% ============================================================== 
ktraj_scaled=reshape(ktraj_scaled,[nFreqEncoding,ninterleavesPerFrame,numFramesToKeep]);
kdata=reshape(kdata,[nFreqEncoding,ninterleavesPerFrame,numFramesToKeep,nChannelsToChoose]);
[coilImages] = coil_sens_map_NUFFT(kdata,ktraj_scaled,N,useGPU);
%coilImages=coilImages(:,:,chh(1:size(chh,2)));
%--------------------------------------------%
% parfor ii=1:size(coilImages,3)
%    coilimages_smoothed(:,:,ii) = medfilt2(abs(coilImages(:,:,ii)));
% end
% tmp1=abs(coilImages(:,:,:));
% tmp2=abs((coilimages_smoothed(:,:,:)));
% [n1,n2,~] =size(tmp1);
% tmp1 = reshape(tmp1,n1*n2,size(coilImages,3));
% tmp2 = reshape(tmp2,n1*n2,size(coilImages,3));
% 
% tmp1=tmp1(:,1:size(coilImages,3))./max(abs(tmp1(:,1:size(coilImages,3))));
% tmp2=tmp2(:,1:size(coilImages,3))./max(abs(tmp2(:,1:size(coilImages,3))));
% noi_img=abs(tmp1-tmp2);
% test = (sum(abs(tmp1),1));
% noise_str=(sum(noi_img,1));
% snr=test./noise_str;
% snr=snr/max(snr(:));
% [~,index] = sort(snr,'descend');
% nChannelsToChoose=length(find(snr>=0.6));%length(index);
% kdata =kdata(:,:,:,index(1:nChannelsToChoose));
% coilImages = coilImages(:,:,index(1:nChannelsToChoose));
%--------------------------------------------%
%kdata=kdata(:,:,:,chh(1:size(chh,2)));
%nChannelsToChoose=size(chh,2);
%% ===============================================================
% Compute coil compresession
% ================================================================
% ind=1:19;
% coilImages=coilImages(:,:,ind);
% kdata=kdata(:,:,:,ind);
% nChannelsToChoose=size(coilImages,3);
kdata=reshape(kdata,[nFreqEncoding*ninterleavesPerFrame*numFramesToKeep,nChannelsToChoose]);

[vkdata,vcoilImages] = combine_coilsv1(kdata,coilImages,0.85);
nChannelsToChoose=size(vcoilImages,3);
kdata=reshape(vkdata,[nFreqEncoding,ninterleavesPerFrame,numFramesToKeep,nChannelsToChoose]);
csm=giveEspiritMaps(reshape(vcoilImages,[size(vcoilImages,1), size(vcoilImages,2), nChannelsToChoose]),0.0025);
coilImages=vcoilImages;

ktraj_scaled=reshape(ktraj_scaled,[nFreqEncoding,ninterleavesPerFrame,numFramesToKeep]);
kdata=reshape(kdata,[nFreqEncoding,ninterleavesPerFrame,numFramesToKeep,nChannelsToChoose]);
%kdata_com=reshape(vkdata,[nFreqEncoding,ninterleavesPerFrame,numFramesToKeep,nChannelsToChoose]);

%% =========================================================
% % Iterative Strategy for non Navigator Data
% %=========================================================
%if navigator==0
    
[kdata_com,ktraj_com] = binSpirals(kdata,ktraj_scaled,600,5,1);
N1 = 300;%ceil(max([max(abs(real(ktraj_com(:)))),max(abs(imag(ktraj_com(:))))]));
csm_lowRes = giveEspiritMapsSmall(coilImages,N1,N1);
ktraj_com = reshape(ktraj_com/N1,[size(ktraj_com,1),size(ktraj_com,2),numFramesToKeep]);
kdata_com = reshape(kdata_com, [size(kdata_com,1),size(ktraj_com,2),numFramesToKeep,nChannelsToChoose]);
%FT_LR= NUFFT(ktraj_com,1,0,0,[N1,N1]);
FT_LR= NUFFT(ktraj_scaled,1,0,0,[N,N]);

WT=Wavelet('Daubechies',4,4);
tic;lowResRecons = l2Recont_v2(kdata_com,FT_LR,csm_lowRes,0.01,N1);toc
%tic;lowResRecons = l2Recont_v2(kdata_com,FT_LR,csm,0.01,N);toc
figure;imagesc(squeeze(abs(lowResRecons(53,:,:))));
lowResRecons=reshape(lowResRecons,[N1*N1,numFramesToKeep]);
[~,~,L]=estimateLapKernelLR(lowResRecons,sigma,lam);
[ xx, L] = iterative_est_laplacian1(L,FT_LR,WT,kdata_com,csm_lowRes, N1,sigma, lam);
xx=reshape(xx,[N1,N1,numFramesToKeep]);
figure;imagesc(squeeze(abs(xx(53,:,:))));

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
%tmp=reshape(tmp,256,256,450);
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
tic; x = solveUV(ktraj_scaled,kdata,csm, V, N, 50,lambdaSmoothness*Sbasis,useGPU,factor);toc
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
save(strcat('datan_','se6v1_',num2str(sl),'.mat'), 'y','-v7.3');
cd(dr2);
%cd './../../../SpiralcodeUVv2/SpiralcodeUV_new';
   % end
%end

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