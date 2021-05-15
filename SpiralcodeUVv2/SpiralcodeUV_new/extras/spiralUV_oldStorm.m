% clear all

addpath(genpath('./../../Data'));
addpath('./csm');
addpath('./nufft_toolbox_cpu');
% addpath(genpath('./../gpuNUFFT'));
% addpath(genpath('./../CUDA'));
% addpath(genpath('./../../../Downloads/gpuNUFFT-master/gpuNUFFT'));
% addpath(genpath('./../../../Downloads/gpuNUFFT-master/CUDA/bin'));
addpath(genpath('./../../../Downloads/gpuNUFFT-master'));



spiralsToDelete=100;
ninterleavesPerFrame=5;
N = 300;
nChannelsToChoose=20;
numFramesToKeep = 100;
%% Reconstruction parameters
maxitCG = 15;
alpha = 1e-5;
tol = 1e-6;
display = 0;
imwidth = N;
useGPU = 'true';
SHRINK_FACTOR = 1.3;
nBasis = 100;
lambdaSmoothness = 0.01;

%%
% % %
% % ==============================================================
% % Load the data
% % ============================================================== 
%Load data
%  load('./../../Data/Spiral_cine_3T_1leave_FB_027_full.mat','kdata','ktraj','dcf');
%  load('./../../Data/cRatioI_027.mat');
pt_no=[27];
 %strr={'UD','VD'};
for p=1:size(pt_no,2)

 str=num2str(pt_no(p));
 load(strcat('Spiral_cine_3T_1leave_FB_0',str,'_full.mat'),'kdata','ktraj','dcf');
 load(strcat('cRatioI_0',str,'.mat'));
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

%% ===============================================================
% Compute coil compresession
% ================================================================
kdata=reshape(kdata,[nFreqEncoding*ninterleavesPerFrame*numFramesToKeep,nChannelsToChoose]);

[vkdata,vcoilImages] = combine_coilsv1(kdata,coilImages,0.85);
nChannelsToChoose=size(vcoilImages,3);
kdata=reshape(vkdata,[nFreqEncoding,ninterleavesPerFrame,numFramesToKeep,nChannelsToChoose]);
csm=giveEspiritMaps(reshape(vcoilImages,[size(vcoilImages,1), size(vcoilImages,2), nChannelsToChoose]));
coilImages=vcoilImages;

%% ==============================================================
% % Compute the weight matrix
% % ============================================================== 

ktraj_scaled=reshape(ktraj_scaled,[nFreqEncoding,ninterleavesPerFrame,numFramesToKeep]);
kdata=reshape(kdata,[nFreqEncoding,ninterleavesPerFrame,numFramesToKeep,nChannelsToChoose]);

[kdata_com,ktraj_com] = binSpirals(kdata,ktraj_scaled,600,5);
N1 = 64;%ceil(max([max(abs(real(ktraj_com(:)))),max(abs(imag(ktraj_com(:))))]));
csm_lowRes = giveEspiritMapsSmall(coilImages,N1,N1);
ktraj_com = reshape(ktraj_com/N1,[size(ktraj_com,1),size(ktraj_com,2),numFramesToKeep]);
kdata_com = reshape(kdata_com, [size(kdata_com,1),size(ktraj_com,2),numFramesToKeep,nChannelsToChoose]);
FT_LR= NUFFT(ktraj_com,1,0,0,[N1,N1]);
tic;lowResRecons = l2Recont_v2(kdata_com,FT_LR,csm_lowRes,0.01);toc
%save(strcat('LRPt',str,'_6_5.mat','lowResRecons'));
% load(strcat('LRPt',str,'_11.mat'));
% lowResRecons=lowResRecons(:,:,1:400);
[~,Si,L]=estimateLapKernelLR(reshape(lowResRecons,[N1*N1,numFramesToKeep]),2.5,0.1);
[~,Sbasis,V]=svd(L);
 V=V(:,end-nBasis+1:end);
 Sbasis=Sbasis(end-nBasis+1:end,end-nBasis+1:end);


ktraj_com=reshape(ktraj_com,[size(ktraj_com,1)*size(ktraj_com,2),numFramesToKeep]);
kdata_com=reshape(kdata_com,[size(kdata_com,1)*size(kdata_com,2),numFramesToKeep,nChannelsToChoose]);
osf = 2; wg = 3; sw = 8;
ktraj_gpu = [real(ktraj_com(:)),imag(ktraj_com(:))]';
FT = gpuNUFFT(ktraj_gpu,ones(size(ktraj_com,1)*size(ktraj_com,2),1),osf,wg,sw,[N1,N1],[],true);

ktraj_scaled=reshape(ktraj_scaled,[nFreqEncoding*ninterleavesPerFrame,numFramesToKeep]);
kdata=reshape(kdata,[nFreqEncoding*ninterleavesPerFrame,numFramesToKeep,nChannelsToChoose]);

for ii=1:10
%[~,Si,L]=estimateLapKernelLRv1(reshape(lowResRecons,[N1*N1,numFramesToKeep]),2.5,1);
%[~,Si,L]=LRrecon(reshape(lowResRecons,[N1*N1,numFramesToKeep]),2.5,1);
%[~,Si1,L1]=estimateLapKernelLR(reshape(lowimg,[N1*N1,numFramesToKeep]),2.5,1);
low_pre=lowResRecons;
Atb = Atb_UV(FT,kdata_com,V,csm_lowRes,true);
Reg = @(x) reshape(reshape(x,[N1*N1,nBasis])*Sbasis,[N1*N1*nBasis,1]);
AtA = @(x) AtA_UV(FT,x,V,csm_lowRes,size(kdata_com,1)) + 1*Reg(x);
tic; x1 = pcg(AtA,Atb(:),1e-5,30);toc;
lowResRecons = reshape(reshape(x1,[N1*N1,nBasis])*V',[N1,N1,numFramesToKeep]);
[~,Si,L]=estimateLapKernelLR(reshape(lowResRecons,[N1*N1,numFramesToKeep]),2.5,0.5);
[~,Sbasis,V]=svd(L);
 V=V(:,end-nBasis+1:end);
 Sbasis=Sbasis(end-nBasis+1:end,end-nBasis+1:end);
if (norm(lowResRecons(:)-low_pre(:))<5e-3) 
    break; 
end

end
%%

%clear kdata_com ktraj_com

    tic; x = solveUV(ktraj_scaled,kdata,csm, V, N, 30,lambdaSmoothness*Sbasis,useGPU);toc
    y = reshape(reshape(x,[N*N,nBasis])*V',[N,N,numFramesToKeep]);
%%
% mkdir './../../Data/Results/LR_finalRecon';
% cd './../../Data/Results/LR_finalRecon';
% save(strcat('res_10Apr_',str,'.mat'), 'y','-v7.3');
% cd './../../../SpiralcodeUVv2/SpiralcodeUV_new';
end
