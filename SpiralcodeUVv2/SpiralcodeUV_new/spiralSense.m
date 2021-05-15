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
spiralsToDelete=40;
ninterleavesPerFrame=4;
N = 300;
nChannelsToChoose=22;
numFramesToKeep = 790;
useGPU = 'true';
SHRINK_FACTOR = 1.3;
nBasis = 12;
lambdaSmoothness = 1;%0.025;
%cRatioI=[1:4,6:14,16:18,21:28,30:34];
%cRatioI=[1,2,3,8,9,11,12,13,14,15,16,17,19,20,21,22,23,24,25,26,29,30];
cRatioI=1:nChannelsToChoose;
sigma=6%[4.5];
lam=3%[2.5];
%%
% % ==============================================================
% % Load the data
% % ============================================================== 
 %load(strcat('Series27.mat'),'kdata','k','dcf');
 dr1='./../../Data/SpiralData_Virg/Phantom/3T_1leave_BH_008_IR_FA10_TBW5_6_fixedSpiral';
 dr2=pwd;
 cd(dr1);
 load('kspaceData.mat');
 cd(dr2);
 
% kdata=kdata(:,1:ninterleavesPerFrame*numFramesToKeep,:);
% kdata = permute(kdata,[1,3,2]);
% k=k(:,1:ninterleavesPerFrame*numFramesToKeep);
% dcf=dcf(:,1:ninterleavesPerFrame*numFramesToKeep);
%%
% %-----------UIOWA Data----------------
% nex=2;
% nsp=1600;
% kdata=squeeze(kdata);
% [npt,~,nch]=size(kdata);
% kdata=reshape(kdata,[npt,nsp,nex,nch]);
% numFrames=floor(nsp/ninterleavesPerFrame);
% kdata=kdata(:,1:numFrames*ninterleavesPerFrame,:,:);
% kdata=reshape(kdata,[npt,numFrames*ninterleavesPerFrame*nex,nch]);
% kdata=permute(kdata,[1,3,2]);
% k=k(:,1:numFrames*ninterleavesPerFrame);dcf=dcf(:,1:numFrames*ninterleavesPerFrame);
% k=repmat(k,[1,nex]);dcf=repmat(dcf,[1,nex]);
% % k=cat(2,k,k,k,k);dcf=cat(2,dcf,dcf,dcf,dcf); 
% % k=cat(2,k,k,k,k);dcf=cat(2,dcf,dcf,dcf,dcf); 

%------------Uni of virg----------%
kdata=kdataUse;
 k=ktrajmUse;
 dcf=dcfUse;
 clear kdataUse ktrajUse dcfUse;
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
ktraj=k;
 %dcf=ones(nFreqEncoding,1);
 
% Reshaping to frames
kdata = permute(kdata,[1,3,2]);
kdata = reshape(kdata,[nFreqEncoding,ninterleavesPerFrame,numFrames,nChannelsToChoose]);
ktraj = reshape(ktraj,[nFreqEncoding,ninterleavesPerFrame,numFrames]);
w = reshape(w,[nFreqEncoding,ninterleavesPerFrame,numFrames]);

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
[coilImages] = coil_sens_map_NUFFT(kdata,ktraj_scaled,N,useGPU);

%% ===============================================================
% Compute coil compresession
% ================================================================
kdata=reshape(kdata,[nFreqEncoding*ninterleavesPerFrame*numFramesToKeep,nChannelsToChoose]);

[vkdata,vcoilImages] = combine_coilsv1(kdata,coilImages,0.85);
nChannelsToChoose=size(vcoilImages,3);
kdata=reshape(vkdata,[nFreqEncoding,ninterleavesPerFrame,numFramesToKeep,nChannelsToChoose]);
csm=giveEspiritMaps(reshape(vcoilImages,[size(vcoilImages,1), size(vcoilImages,2), nChannelsToChoose]),0.003);
coilImages=vcoilImages;

ktraj_scaled=reshape(ktraj_scaled/N,[nFreqEncoding,ninterleavesPerFrame,numFramesToKeep]);
kdata=reshape(kdata,[nFreqEncoding,ninterleavesPerFrame,numFramesToKeep,nChannelsToChoose]);
%% --------------------------------------------------------------
% % SENSE Reconstruction
% % -------------------------------------------------------------
FT= NUFFT(ktraj_scaled,1,0,0,[N,N]);
tic;y_o = l2Recont_v2(kdata,FT,csm,0.01,N);toc



%% ==============================================================
% % Compute the weight matrix
% % ============================================================= 
%no_ch=size(csm,3);
%Nav=permute((kdata(:,1,:,:)),[1,2,4,3]);
%Nav=Nav(:,:,1:264);
% Nav=reshape(permute(Nav,[1,3,2]),[nFreqEncoding*2,floor(numFramesToKeep/2),nChannelsToChoose]);
% Nav=permute(Nav,[1,3,2]);
% for ii=1:size(sigma,2)
%     for jj=1:size(lam,2)
% [~,~,L]=estimateLapKernelLR(reshape(Nav,[nFreqEncoding*no_ch,numFramesToKeep]),sigma(ii),lam(jj));
% [~,Sbasis,V]=svd(L);
% V=V(:,end-nBasis+1:end);
% Sbasis=Sbasis(end-nBasis+1:end,end-nBasis+1:end);
% %% ==============================================================
% % % Final Reconstruction
% % % ============================================================= 
% factor=0;
% ktraj_scaled=reshape(ktraj_scaled,[nFreqEncoding*ninterleavesPerFrame,numFramesToKeep]);
% kdata=reshape(kdata,[nFreqEncoding*ninterleavesPerFrame,numFramesToKeep,nChannelsToChoose]);
% tic; x = solveUV(ktraj_scaled,kdata,csm, V, N, 50,lambdaSmoothness*Sbasis,useGPU,factor);toc
% % for i=1:1
% %     factor=(abs(x)+mean(abs(x(:))).*4);
% %     tic; x = solveUV(ktraj_scaled,kdata,csm, V, N, 40,lambdaSmoothness*Sbasis,useGPU,factor);toc
% % end
% 
% y = reshape(reshape(x,[N*N,nBasis])*V',[N,N,numFramesToKeep]);

%% ==============================================================
% % Save and Display results
% % ============================================================= 

%for i=1:530;imagesc(fliplr(flipud(abs(y(:,:,i))))); pause(0.1); colormap gray;end
%for i=1:250;imagesc(((abs(y(:,:,i))))); pause(0.1); colormap gray; end
%clear kdata csm V ;
%mkdir './../../Data/SpiralData_24Aug18_UIOWA/Series17/';
%cd './../../Data/SpiralData_Virg/Phantom/3T_';
cd(dr1);
save(strcat('sense_',num2str(lambdaSmoothness),'.mat'), 'y_o','-v7.3');
%cd './../../../SpiralcodeUVv2/SpiralcodeUV_new'
    %end
%end


% for i=1:12;colormap gray;
%     subplot(3,4,i);imagesc(((abs(X(:,:,i)))));
% end
% figure;hold on;
%  for i=1:12
%      subplot(4,3,i);
%      plot(V(:,end+1-i));
%  end