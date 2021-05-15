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



spiralsToDelete=50;
ninterleavesPerFrame=25;
N = 400;256%300;
nChannelsToChoose=6;
numFramesToKeep = 478;
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
%chh=1:15;
%chh=[1,3,5,9,12];
chh=[3,5,7,9,12,15];

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
lam=[0.1];%0.1,0.001,0.05];
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
numFrames=478;
nFreqEncoding=256;
ktraj=k;
kdata = reshape(kdata,[nFreqEncoding,ninterleavesPerFrame,numFrames,nChannelsToChoose]);
ktraj = reshape(ktraj,[nFreqEncoding,ninterleavesPerFrame,numFrames]);
% Keeping only numFramesToKeep

kdata = kdata(:,:,1:numFramesToKeep,:);
%load('LRdata.mat');
%kdata=knew;
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
[coilImages] = coil_sens_map_NUFFT(kdata(:,:,1:100,:),ktraj_scaled(:,:,1:100),N,useGPU);
%load(strcat('coil_img_',str,'.mat'));
%csm=csm1;
%% ===============================================================
% Compute coil compresession
% ================================================================
%load('vkdata.mat');
%load('vcoilImg_27.mat');
 kdata=reshape(kdata,[nFreqEncoding*ninterleavesPerFrame*numFramesToKeep,nChannelsToChoose]);
 [vkdata,vcoilImages] = combine_coilsv1(kdata,coilImages,0.8);
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

[kdata_com,ktraj_com] = binSpirals(kdata,ktraj_scaled,32,25,0);
%kdata_com=kdata;%((nFreqEncoding/4)+1:(nFreqEncoding*0.75),:,:,:);
%ktraj_com=ktraj_scaled;%((nFreqEncoding/4)+1:(nFreqEncoding*0.75),:,:);
 N1 = 32;%ceil(max([max(abs(real(ktraj_com(:)))),max(abs(imag(ktraj_com(:))))]));
csm_lowRes = giveEspiritMapsSmall(coilImages,N1,N1);
ktraj_com = reshape(ktraj_com/N1,[size(ktraj_com,1),size(ktraj_com,2),numFramesToKeep]);
kdata_com = reshape(kdata_com, [size(kdata_com,1),size(ktraj_com,2),numFramesToKeep,nChannelsToChoose]);
%kdata_com=kdata_com(:,:,:,1:3);
w=sqrt(abs(ktraj_com));w=w(:,:,1);
FT_LR= NUFFT(ktraj_com,w,0,0,[N1,N1]);%csm_lowRes=csm_lowRes(:,:,1:3);
tic;lowResRecons = l2Recont_v2(kdata_com,FT_LR,csm_lowRes,0.01,N1);toc
% roi=lowResRecons(5:260,65:320,:);
% for ij=1:2;
% knew(:,:,:,ij)=FT_LR*(bsxfun(@times,roi,csm_lowRes(:,:,ij)));
% end
% %load('LR27.mat');
% %save(strcat('LR',str,'.mat','lowResRecons'));
% %load(strcat('LRPt',str,'_11.mat'));
% %lowResRecons=lowResRecons(:,:,1:380);
% %load(strcat('temp_LR',str,'_0.1.mat'));
% %lowResRecons=tmp;
 lowResRecons=reshape(lowResRecons,[N1*N1,numFramesToKeep]);
for ij=1:1
     if ij>=2
        lam(jj)=lam(jj)-0.045;
    end
[~, tmp, ~] = est_laplacian_LR4(lowResRecons,FT_LR,kdata_com,csm_lowRes,N1, 4.5, lam(jj));
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
 lamb=[0.02];
for i=1:size(lamb,2)
tic; x = solveUV(ktraj_scaled,kdata,csm, V, N, 50,lamb(i)*Sbasis,useGPU,0);toc
y = reshape(reshape(x,[N*N,nBasis])*V',[N,N,numFramesToKeep]);
%%
%clear kdata csm V ;
save(strcat('testrecv5181',num2str(lam(jj)),'.mat'),'y','lowResRecons');
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
clear kdata ktraj; nChannelsToChoose=10;
end



% % 
% for i=1:15;colormap gray;
%     subplot(3,5,i);imagesc((flipud(abs(coilImages(:,:,i)))));
% end
% 
% subplot(121);imagesc(abs(flipud(mean(coilImages(25:220,75:160,:),3))));
% newcoilImages=zeros(256,256,5);colormap gray
% newcoilImages=coilImages(25:220,75:160,[1,3,5,9,12]);
% subplot(122);imagesc(abs(flipud(mean(newcoilImages,3))));
% for i=1:250;imagesc((flipud(abs(y(100:300,100:300,i))))); pause(0.05); colormap gray;end
