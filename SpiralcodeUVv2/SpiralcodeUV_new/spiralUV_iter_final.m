% clear all
%%
addpath(genpath('./../../Data'));
addpath('./csm');
addpath('./nufft_toolbox_cpu');
addpath(genpath('./gpuNUFFT'));
addpath(genpath('./CUDA'));


%% Reconstruction Parameters
spiralsToDelete=100;
ninterleavesPerFrame=6;
N = 340;
nChannelsToChoose=8;
numFramesToKeep = 500;
useGPU = 'true';
SHRINK_FACTOR = 1.0;
nBasis = 30;
lambdaSmoothness = 0.025;
cRatioI=1:nChannelsToChoose;
sigma=[4.5];
lam=[0.1];
%%
% % ==============================================================
% % Load the data
% % ============================================================== 
 dr1='./../../Data/SpiralData_2Oct18_UIOWA/Series10/';
 dr2=pwd;
 cd(dr1);
 load('Series10.mat');
 cd(dr2);

 %strr={'UD','VD'};
for p=1:size(pt_no,2)

 kdata=squeeze(kdata);
 
 
 [nFreqEncoding,nCh,numberSpirals]=size(kdata);
 numFrames=floor((numberSpirals-spiralsToDelete)/ninterleavesPerFrame);
 

 kdata=kdata(:,cRatioI(1:nChannelsToChoose),spiralsToDelete+1:numberSpirals);
 ktraj=cell2mat(ktraj);
 ktraj = ktraj(:,spiralsToDelete+1:numberSpirals);
 %dcf=ones(nFreqEncoding,1);
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
[csm1,coilImages] = coil_sens_map_NUFFT(kdata,ktraj_scaled,N,useGPU);
%load(strcat('coil_img_',str,'.mat'));
%% ===============================================================
% Compute coil compresession
% ================================================================
%load('vkdata.mat');
%load('vcoilImg_27.mat');
kdata=reshape(kdata,[nFreqEncoding*ninterleavesPerFrame*numFramesToKeep,nChannelsToChoose]);

[vkdata,vcoilImages] = combine_coilsv1(kdata,coilImages,0.85);
 nChannelsToChoose=size(vcoilImages,3);
 kdata=reshape(vkdata,[nFreqEncoding,ninterleavesPerFrame,numFramesToKeep,nChannelsToChoose]);
csm=giveEspiritMaps(reshape(vcoilImages,[size(vcoilImages,1), size(vcoilImages,2), nChannelsToChoose]));
coilImages=vcoilImages;
%save('vcoilImg_27.mat','coilImages','csm','-v7.3');


%% ==============================================================
% % Compute the weight matrix
% % ============================================================== 
for jj=1:size(lam,2)

ktraj_scaled=reshape(ktraj_scaled,[nFreqEncoding,ninterleavesPerFrame,numFramesToKeep]);
kdata=reshape(kdata,[nFreqEncoding,ninterleavesPerFrame,numFramesToKeep,nChannelsToChoose]);

[kdata_com,ktraj_com] = binSpirals(kdata,ktraj_scaled,600,7);
 N1 = 64;%ceil(max([max(abs(real(ktraj_com(:)))),max(abs(imag(ktraj_com(:))))]));
csm_lowRes = giveEspiritMapsSmall(coilImages,N1,N1);
ktraj_com = reshape(ktraj_com/N1,[size(ktraj_com,1),size(ktraj_com,2),numFramesToKeep]);
kdata_com = reshape(kdata_com, [size(kdata_com,1),size(ktraj_com,2),numFramesToKeep,nChannelsToChoose]);
FT_LR= NUFFT(ktraj_com,1,0,0,[N1,N1]);
% tic;lowResRecons = l2Recont_v2(kdata_com,FT_LR,csm_lowRes,0.01);toc
% %load('LR27.mat');
% %save(strcat('LR',str,'.mat','lowResRecons'));
% %load(strcat('LRPt',str,'_11.mat'));
% %lowResRecons=lowResRecons(:,:,1:380);
% lowResRecons=reshape(lowResRecons,[N1*N1,numFramesToKeep]);
% for ij=1:1
%      if ij>=3
%         lam(jj)=lam(jj)*20;
%     end
% [~, tmp, L] = est_laplacian_LR3(lowResRecons,FT_LR,kdata_com,csm_lowRes, 4.5, lam(jj));
% %lowResRecons=tmp;
% end
load(strcat('temp_LR',str,'.mat'));
lowResRecons=tmp;
%[~, tmp, ~] = est_laplacian_LR3(lowResRecons,FT_LR,kdata_com,csm_lowRes, 4.5, lam(jj));
%save(strcat('LR_',str,'.mat'),'tmp');
%lowResRecons=tmp;
[~,~,L]=estimateLapKernelLR(reshape(lowResRecons,[N1*N1,numFramesToKeep]),sigma(1),lam(jj));
[~,Sbasis,V]=svd(L);
V=V(:,end-nBasis+1:end);
Sbasis=Sbasis(end-nBasis+1:end,end-nBasis+1:end);
% 
% 
% 
 ktraj_scaled=reshape(ktraj_scaled,[nFreqEncoding*ninterleavesPerFrame,numFramesToKeep]);
 kdata=reshape(kdata,[nFreqEncoding*ninterleavesPerFrame,numFramesToKeep,nChannelsToChoose]);
% 
% %%
% 
% %clear kdata_com ktraj_com L csm_lowRes;
% 
 lamb=[0,0.001,0.1];
for i=1:size(lamb,2)
tic; x = solveUV(ktraj_scaled,kdata,csm, V, N, 50,lamb(i)*Sbasis,useGPU);toc
y = reshape(reshape(x,[N*N,nBasis])*V',[N,N,numFramesToKeep]);
%%
%clear kdata csm V ;
mkdir './../../Data/Results/res_June8';
cd './../../Data/Results/res_June8';
save(strcat('res0.03_',str,'_',num2str(lamb(i)),'_',num2str(lam(jj)),'.mat'), 'y','-v7.3');
cd './../../../SpiralcodeUVv2/SpiralcodeUV_new';
end
clear lowResRecons;
end
end
