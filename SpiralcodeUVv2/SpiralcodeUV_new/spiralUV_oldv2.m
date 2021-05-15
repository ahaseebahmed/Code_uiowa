% clear all

addpath(genpath('./../../Data/SpiralData_UVirginia/Spiral_3T'));
addpath('./csm');
addpath('./nufft_toolbox_cpu');
addpath('./utils');
addpath('./Utils');
addpath('./extras');
% addpath(genpath('./../gpuNUFFT'));
% addpath(genpath('./../CUDA'));
% addpath(genpath('./../../../Downloads/gpuNUFFT-master/gpuNUFFT'));
% addpath(genpath('./../../../Downloads/gpuNUFFT-master/CUDA/bin'));
%addpath(genpath('./../../../Downloads/gpuNUFFT-master'));
addpath(genpath('./gpuNUFFT'));
addpath(genpath('./CUDA'));



spiralsToDelete=100;
ninterleavesPerFrame=4;
N = 320;
nChannelsToChoose=34;
numFramesToKeep = 425;
%% Reconstruction parameters
maxitCG = 15;
alpha = 1e-5;
tol = 1e-6;
display = 0;
imwidth = N;
useGPU = 'true';
SHRINK_FACTOR = 1.5;
nBasis = 30;
lambdaSmoothness = 0.0;
cRatioI=1:nChannelsToChoose;
%%
% % %
% % ==============================================================
% % Load the data
% % ============================================================== 
%Load data
%  load('./../../Data/Spiral_cine_3T_1leave_FB_027_full.mat','kdata','ktraj','dcf');
%  load('./../../Data/cRatioI_027.mat');
pt_no=[27];
sigma=[4.5];
lam=[0.1];
sigma2=[0.01];
lam2=[0.0];
 %strr={'UD','VD'};
for p=1:size(pt_no,2)

 str=num2str(pt_no(p));
 load(strcat('./../../Data/SpiralData_UVirginia/Spiral_3T/3T_1leave_FB_057_mid_fixedSpiral/kspaceData.mat'),'kdataUse','ktrajmUse');
 %load(strcat('cRatioI_0',str,'.mat'));
 kdata=squeeze(kdataUse);
 ktraj=squeeze(ktrajmUse);
 
 [nFreqEncoding,nCh,numberSpirals]=size(kdata);
 numFrames=floor((numberSpirals-spiralsToDelete)/ninterleavesPerFrame);
 

 kdata=kdata(:,cRatioI(1:nChannelsToChoose),spiralsToDelete+1:numberSpirals);
 %ktraj=cell2mat(ktraj);
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
[coilImages] = coil_sens_map_NUFFT(kdata,ktraj_scaled,N,useGPU);

%% ===============================================================
% Compute coil compresession
% ================================================================
kdata=reshape(kdata,[nFreqEncoding*ninterleavesPerFrame*numFramesToKeep,nChannelsToChoose]);

[vkdata,vcoilImages] = combine_coilsv1(kdata,coilImages,0.85);
nChannelsToChoose=size(vcoilImages,3);
kdata=reshape(vkdata,[nFreqEncoding,ninterleavesPerFrame,numFramesToKeep,nChannelsToChoose]);
csm=giveEspiritMaps(reshape(vcoilImages,[size(vcoilImages,1), size(vcoilImages,2), nChannelsToChoose]),0.005);
coilImages=vcoilImages;

%% ==============================================================
% % Compute the weight matrix
% % ============================================================== 
for jj=1:size(lam,2)
    for ii=1:size(sigma2,2)
        for kk=1:size(sigma,2) %#ok<*ALIGN>
   

ktraj_scaled=reshape(ktraj_scaled,[nFreqEncoding,ninterleavesPerFrame,numFramesToKeep]);
kdata=reshape(kdata,[nFreqEncoding,ninterleavesPerFrame,numFramesToKeep,nChannelsToChoose]);

[kdata_com,ktraj_com] = binSpirals(kdata,ktraj_scaled,600,9,1);
 N1 = 64;%ceil(max([max(abs(real(ktraj_com(:)))),max(abs(imag(ktraj_com(:))))]));
csm_lowRes = giveEspiritMapsSmall(coilImages,N1,N1);
ktraj_com = reshape(ktraj_com/N1,[size(ktraj_com,1),size(ktraj_com,2),numFramesToKeep]);
kdata_com = reshape(kdata_com, [size(kdata_com,1),size(ktraj_com,2),numFramesToKeep,nChannelsToChoose]);
FT_LR= NUFFT(ktraj_com,1,0,0,[N1,N1]);
tic;lowResRecons = l2Recont_v2(kdata_com,FT_LR,csm_lowRes,0.01,N1);toc
%save(strcat('LRPt',str,'_6_5.mat','lowResRecons'));
%load(strcat('LRPt',str,'_11.mat'));
%lowResRecons=lowResRecons(:,:,1:380);
%load(strcat('/Users/ahhmed/Codes/Data/Results/TMP/res_',str,'.mat'));
[~,~,L]=estimateLapKernelLR(reshape(lowResRecons,[N1*N1,numFramesToKeep]),sigma(kk),lam(jj));
[~,Sbasis,V]=svd(L);
V=V(:,end-nBasis+1:end);
Sbasis=Sbasis(end-nBasis+1:end,end-nBasis+1:end);

% 
ktraj_com=reshape(ktraj_com,[size(ktraj_com,1)*size(ktraj_com,2),numFramesToKeep]);
kdata_com=reshape(kdata_com,[size(kdata_com,1)*size(kdata_com,2),numFramesToKeep,nChannelsToChoose]);
osf = 2; wg = 3; sw = 8;
ktraj_gpu = [real(ktraj_com(:)),imag(ktraj_com(:))]';
FT = gpuNUFFT(ktraj_gpu,ones(size(ktraj_com,1)*size(ktraj_com,2),1),osf,wg,sw,[N1,N1],[],true);

ktraj_scaled=reshape(ktraj_scaled,[nFreqEncoding*ninterleavesPerFrame,numFramesToKeep]);
kdata=reshape(kdata,[nFreqEncoding*ninterleavesPerFrame,numFramesToKeep,nChannelsToChoose]);

for i=1:10
%[~,Si,L]=estimateLapKernelLRv1(reshape(lowResRecons,[N1*N1,numFramesToKeep]),2.5,1);
%[~,Si,L]=LRrecon(reshape(lowResRecons,[N1*N1,numFramesToKeep]),2.5,1);
%[~,Si1,L1]=estimateLapKernelLR(reshape(lowimg,[N1*N1,numFramesToKeep]),2.5,1);
low_pre=lowResRecons;
Atb = Atb_UV(FT,kdata_com,V,csm_lowRes,N1,true);
Reg = @(x) reshape(reshape(x,[N1*N1,nBasis])*Sbasis,[N1*N1*nBasis,1]);
AtA = @(x) AtA_UV(FT,x,V,csm_lowRes,N1,size(kdata_com,1)) + sigma2(ii)*Reg(x);
tic; x1 = pcg(AtA,Atb(:),1e-5,50);toc;
lowResRecons = reshape(reshape(x1,[N1*N1,nBasis])*V',[N1,N1,numFramesToKeep]);
[~,~,L]=estimateLapKernelLR(reshape(lowResRecons,[N1*N1,numFramesToKeep]),sigma(kk),lam(jj));
[~,Sbasis,V]=svd(L);
V=V(:,end-nBasis+1:end);
Sbasis=Sbasis(end-nBasis+1:end,end-nBasis+1:end);
if (norm(lowResRecons(:)-low_pre(:))<1e-3) 
    break; 
end

end
%%

%clear kdata_com ktraj_com L csm_lowRes;
% for ll=1:size(lam2,2)
 tic; x = solveUV(ktraj_scaled,kdata,csm, V, N, 60,0.01*Sbasis,useGPU,0);toc
 y = reshape(reshape(x,[N*N,nBasis])*V',[N,N,numFramesToKeep]);
%%
%clear kdata csm V ;
mkdir './../../Data/Results/res_20Jan';
cd './../../Data/Results/res_20Jan';
save(strcat('Sense_res0_','57_',num2str(sigma2(ii)),'_',num2str(sigma(kk)),'_',num2str(lam(jj)),'.mat'),'y','lowResRecons','-v7.3');
cd './../../../SpiralcodeUVv2/SpiralcodeUV_new';

            end
clear lowResRecons;
        end
    end
end

%end