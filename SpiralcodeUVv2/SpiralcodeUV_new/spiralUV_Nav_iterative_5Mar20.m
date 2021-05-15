clear all

addpath('./csm');
addpath('./utils');
addpath('./Utils');
addpath('./extras');
addpath('./nufft_toolbox_cpu');
addpath(genpath('./gpuNUFFT'));
%addpath(genpath('./../../Wavelab850'));
addpath(genpath('/Users/ahhmed/Codes/Data/UIUC'));

%% Reconstruction parameters
spiralsToDelete=100%60;
ninterleavesPerFrame=5;
N = 400;
nBasis=30;
numFramesToKeep =100;
chh=1:34;
maxitCG = 15;
useGPU = true;
SHRINK_FACTOR = 1;
nChannelsToChoose=30;
framesToDelete=0;
sigma=4.5;
lam=0.1;
lambdaSmoothness=0.02;
%%
% % ==============================================================
% % Load the data
% % ============================================================== 
%% =========================================================
% % Iterative Strategy for non Navigator Data
% %=========================================================
%if navigator==0
  
kdata1=reshape(kdata1,[2496,11*256]);
k=reshape(k,[2496,11*256]);
j=1;
for i=1:256
    
    kdata(:,:,i)=kdata1(:,(j-1)+1:j+11);
    k1(:,:,i)=k(:,(j-1)+1:j+11);
    j=j+10;
    
end

%kdata_com=kdata;
ktraj=k1/(2*max(abs(k1(:))));
N1=200;
FT= NUFFT(ktraj,1,0,0,[N1,N1]);
tmp=FT*x;
%WT=Wavelet('Daubechies',4,4);
%load('tmp_L.mat');
csm_lowRes=1;
[kdata_com,ktraj_com] = binSpirals(tmp,ktraj,2496,11,1);
ktraj_com=ktraj_com/(2*max(abs(ktraj_com(:))));
FT_LR= NUFFT(ktraj_com,1,0,0,[200,200]);

tic;lowResRecons = l2Recont_v2(kdata_com,FT_LR,csm_lowRes,0.01,200);toc
lowResRecons=reshape(lowResRecons,[50*50,numFramesToKeep]);
%  q= fft(lowResRecons,[],2);
% q(:,numFramesToKeep/2-numFramesToKeep/4-150:numFramesToKeep/2+numFramesToKeep/4+150)=0;
% im_nav_smoothed = ifft(q,[],2);
%[ ~,X,~]=est_laplacian_rev(lowResRecons,FT_LR,kdata_com,csm_lowRes,N1, sigma, lam);
%[~, tmp, L] = est_laplacian_LR3(lowResRecons,FT_LR,kdata_com,csm_lowRes,N1, sigma, lam);
WT=1;
[~,~,L]=estimateLapKernelLR(lowResRecons,sigma,lam);
[ X, ~] = iterative_est_laplacian1(L,FT,WT,kdata_com,csm_lowRes, N1,sigma, lam);
  q= fft(X,[],2);
 q(:,numFramesToKeep/2-numFramesToKeep/4-10:numFramesToKeep/2+numFramesToKeep/4+10)=0;
 X1 = ifft(q,[],2);
% cd(dr1);
% load(strcat('resn_14_',num2str(sl),'.mat'));
% cd(dr2);
%X1 = reshape(x,[400*400,nBasis])*V';
%   q= fft(X1,[],2);
%  q(:,numFramesToKeep/2-numFramesToKeep/4-30:numFramesToKeep/2+numFramesToKeep/4+30)=0;
%  X1 = ifft(q,[],2);

[~,~,L]=estimateLapKernelLR(X,sigma,0.001);
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
kdata=tmp;
ktraj_scaled=reshape(ktraj,[nFreqEncoding*ninterleavesPerFrame,numFramesToKeep]);
kdata=reshape(kdata,[nFreqEncoding*ninterleavesPerFrame,numFramesToKeep,nChannelsToChoose]);
tic; x1 = solveUV(ktraj_scaled,kdata,csm, V, N, 50,lambdaSmoothness*Sbasis,useGPU,factor);toc
% for i=1:1
%     factor=(abs(x)+mean(abs(x(:))).*10);
%     tic; x = solveUV(ktraj_scaled,kdata,csm, V, N, 30,lambdaSmoothness*Sbasis,useGPU,factor);toc
% end

y = reshape(reshape(x1,[N*N,nBasis])*V',[N,N,numFramesToKeep]);

%% ==============================================================
% % Save and Display results
% % ============================================================= 

%for i=1:530;imagesc(fliplr(flipud(abs(y(:,:,i))))); pause(0.1); colormap gray;end
%for i=1:250;imagesc(((abs(y(:,:,i))))); pause(0.1); colormap gray; end
%clear kdata csm V ;
%mkdir './../../Data/SpiralData_24Aug18_UIOWA/Series17/';
%cd './../../Data/SpiralData_Virg/Phantom/3T_';
% cd(dr1);
% save(strcat('resn_16_',num2str(sl),'.mat'), 'x','V','-v7.3');
% cd(dr2);
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