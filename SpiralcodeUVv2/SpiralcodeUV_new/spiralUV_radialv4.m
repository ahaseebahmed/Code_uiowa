 %clear all

addpath(genpath('./../../Data'));
addpath('./csm');
addpath('./Utils');
addpath('./extras');
addpath('./utils');
addpath('./nufft_toolbox_cpu');
% addpath(genpath('./../gpuNUFFT'));
% addpath(genpath('./../CUDA'));
% addpath(genpath('./../../../Downloads/gpuNUFFT-master/gpuNUFFT'));
% addpath(genpath('./../../../Downloads/gpuNUFFT-master/CUDA/bin'));
%addpath(genpath('./../../../Downloads/gpuNUFFT-master'));
addpath(genpath('./gpuNUFFT'));
addpath(genpath('./CUDA'));



spiralsToDelete=75;
ninterleavesPerFrame=15;
N = 400%256%300;
nChannelsToChoose=6;
numFramesToKeep = 795;
%% Reconstruction parameters
maxitCG = 15;
alpha = 1e-5;
tol = 1e-6;
display = 0;
imwidth = N;
useGPU = 'true';
SHRINK_FACTOR =1;
nBasis = 30;
lambdaSmoothness = 0.0;
%chh=[1,3:4,6:8,10:11,13:14];
%chh=[1,2,3,4,6,15];
chh=[3,5,11,9,12,15];
%chh=[9,12];

%chh=1:15;
slc=[2:5,8,9,10];
%%
% % %
% % ==============================================================
% % Load the data
% % ============================================================== 
%Load data
%  load('./../../Data/Spiral_cine_3T_1leave_FB_027_full.mat','kdata','ktraj','dcf');
%  load('./../../Data/cRatioI_027.mat');
%pt_no=[11,28];
sigma=4.5;
lam=[0.025];%,0.1,0.001,0.05];
 %strr={'UD','VD'};
%for p=1:size(pt_no,2)

 %str=num2str(pt_no(p));
 %load(strcat('Spiral_cine_3T_1leave_FB_0',str,'_full.mat'),'kdata','ktraj','dcf');
 %load(strcat('cRatioI_0',str,'.mat'));
for p=7:size(slc,2)
 load('golden.mat');
 kdata=squeeze(ksp(:,chh,:,slc(p)));
  load ('traj.mat');

%  
%  
[~,nCh,numberSpirals]=size(kdata);
  numFrames=floor((numberSpirals-spiralsToDelete)/ninterleavesPerFrame);
numberSpirals=numFrames*ninterleavesPerFrame+spiralsToDelete;
% 
  kdata=kdata(:,:,spiralsToDelete+1:numberSpirals);
%  ktraj=cell2mat(ktraj);
  k = k(:,spiralsToDelete+1:numberSpirals);
%  %dcf=ones(nFreqEncoding,1);
% % Reshaping to frames
% 
 kdata = permute(kdata,[1,3,2]);
%% FOR Penn data
numFrames=795;
nFreqEncoding=256;
ktraj=k;
kdata = reshape(kdata,[nFreqEncoding,ninterleavesPerFrame,numFrames,nChannelsToChoose]);
ktraj = reshape(ktraj,[nFreqEncoding,ninterleavesPerFrame,numFrames]);
% Keeping only numFramesToKeep

kdata = kdata(:,:,1:numFramesToKeep,:);
ktraj = ktraj(:,:,1:numFramesToKeep);
%save data kdata ktraj dcf
%%

[~,numberSpirals,~,~]=size(kdata);
ktraj_scaled =  SHRINK_FACTOR*ktraj;
%% ==============================================================
% Compute the coil sensitivity map
% ============================================================== 
ktraj_scaled=reshape(ktraj_scaled,[nFreqEncoding,ninterleavesPerFrame,numFramesToKeep]);
kdata=reshape(kdata,[nFreqEncoding,ninterleavesPerFrame,numFramesToKeep,nChannelsToChoose]);
[coilImages] = coil_sens_map_NUFFT(kdata(:,:,1:200,:),ktraj_scaled(:,:,1:200),N,useGPU);
%load(strcat('coil_img_',str,'.mat'));
%csm=csm1;
%% ===============================================================
% Compute coil compresession
% ================================================================
%load('vkdata.mat');
%load('vcoilImg_27.mat');
 kdata=reshape(kdata,[nFreqEncoding*ninterleavesPerFrame*numFramesToKeep,nChannelsToChoose]);
 [vkdata,vcoilImages] = combine_coilsv1(kdata,coilImages,0.80);
%  chh=[3,4];vcoilImages=vcoilImages(:,:,chh);vkdata=vkdata(:,chh);
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

[kdata_com,ktraj_com] = binSpirals(kdata,ktraj_scaled,64,19,0);
%kdata_com=kdata;%((nFreqEncoding/4)+1:(nFreqEncoding*0.75),:,:,:);
%ktraj_com=ktraj_scaled;%((nFreqEncoding/4)+1:(nFreqEncoding*0.75),:,:);
 N1 = 64;%ceil(max([max(abs(real(ktraj_com(:)))),max(abs(imag(ktraj_com(:))))]));
csm_lowRes = giveEspiritMapsSmall(coilImages,N1,N1);
ktraj_com = reshape(ktraj_com/N1,[size(ktraj_com,1),size(ktraj_com,2),numFramesToKeep]);
kdata_com = reshape(kdata_com, [size(kdata_com,1),size(ktraj_com,2),numFramesToKeep,nChannelsToChoose]);
kdata_com=kdata_com(:,:,:,1);
w=sqrt(abs(ktraj_com));w=w(:,:,1);
FT_LR= NUFFT(ktraj_com,w,0,0,[N1,N1]);csm_lowRes=csm_lowRes(:,:,1);
tic;lowResRecons = l2Recont_v2(kdata_com,FT_LR,csm_lowRes,0.01,N1);toc
% %load('LR27.mat');
% %save(strcat('LR',str,'.mat','lowResRecons'));
% %load(strcat('LRPt',str,'_11.mat'));
% %lowResRecons=lowResRecons(:,:,1:380);
% %load(strcat('temp_LR',str,'_0.1.mat'));
% %lowResRecons=tmp;
 lowResRecons=reshape(lowResRecons,[N1*N1,numFramesToKeep]);
for ij=1:1
     if ij>=3
        lam(jj)=lam(jj);
    end
[~, tmp, ~] = est_laplacian_LR3(lowResRecons,FT_LR,kdata_com,csm_lowRes,N1, 4.5, lam(jj));
lowResRecons=tmp;
end
% %load(strcat('temp_LR',str,'_0.1.mat'));
% %load('tempp2221.mat');
% lowResRecons=tmp;
%lam=0.1;
[~,Si,L]=estimateLapKernelLR(reshape(lowResRecons,[N1*N1,numFramesToKeep]),sigma(1),0.1);
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
 lamb=[0.025];
for i=1:size(lamb,2)
tic; x = solveUV(ktraj_scaled,kdata,csm, V, N, 70,lamb(i)*Sbasis,useGPU);toc
y = reshape(reshape(x,[N*N,nBasis])*V',[N,N,numFramesToKeep]);
%%
%clear kdata csm V ;
save(strcat('testrecv515',num2str(slc(p)),'.mat'),'y','lowResRecons');
% xx=abs(squeeze(x(:,35:192,:)));
% yy=abs(squeeze(y(29:228,63:220,:)));
%  xx=(xx-min(xx(:)))/(max(xx(:))-min(xx(:)));
%  yy=(yy-min(yy(:)))/(max(yy(:))-min(yy(:)));
% tmpp(i)=SNR_3D(xx,yy)
 

% mkdir './../../Data/Results/res_June9';
% cd './../../Data/Results/res_June9';
% save(strcat('res0.001_',str,'_',num2str(lamb(i)),'_',num2str(lam(jj)),'.mat'), 'y','tmp','-v7.3');
% cd './../../../SpiralcodeUVv2/SpiralcodeUV_new';
end
clear lowResRecons;
end
clear kdata ktraj; nChannelsToChoose=6;
end
