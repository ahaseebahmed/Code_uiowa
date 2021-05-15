clear all

addpath('./../');
addpath('./csm');
addpath('./nufft_toolbox_cpu');
addpath(genpath('./../../gpuNUFFT'));

spiralsToDelete=60;
ninterleavesPerFrame=15;
N = 256;
nChannels=6;
nBasis=30;
Nonfft = true;
numFramesToKeep = 700;
chh=[3,7,5,9,12,15];
cRatioI=1:nChannels;
%% Reconstruction parameters
maxitCG = 15;
alpha = 1e-5;
tol = 1e-6;
display = 0;
imwidth = N;
useGPU = true;
SHRINK_FACTOR = 1.0;
%
% % %
% % ==============================================================
% % Load the data
% % ============================================================== 
%Load data
 load('golden.mat');
 %load('traj.mat');
 ksp = squeeze(ksp(:,chh,:,11));
 %%
  kdata=ksp;
 
 [nFreqEncoding,nCh,numberSpirals]=size(kdata);
 delta = 0.2; % shift in k-space for eddy current adjustment
 k = giveGoldenAngleTraj(nFreqEncoding,numberSpirals,delta);
 
 numFrames=floor((numberSpirals-spiralsToDelete)/ninterleavesPerFrame);
 

 kdata=kdata(:,cRatioI(1:nChannels),spiralsToDelete+1:numberSpirals);
 ktraj = k(:,spiralsToDelete+1:numberSpirals);
 
 shift = -0+0*1i;
 modulation = exp(1i*real(shift*ktraj)/N);
 
 [nFreqEncoding,~,numberSpirals]=size(kdata);

 kdata = bsxfun(@times,kdata,reshape(modulation,[nFreqEncoding,1,numberSpirals]));
% Reshaping to frames

kdata = permute(kdata,[1,3,2]);
kdata = reshape(kdata,[nFreqEncoding,ninterleavesPerFrame,numFrames,nChannels]);
ktraj = reshape(ktraj,[nFreqEncoding,ninterleavesPerFrame,numFrames]);
% Keeping only numFramesToKeep

kdata = kdata(:,:,1:numFramesToKeep,:);
ktraj = ktraj(:,:,1:numFramesToKeep);
%save data kdata ktraj dcf
%%

[nFreqEncoding,numberSpirals,~,~]=size(kdata);
ktraj_scaled =  SHRINK_FACTOR*ktraj;

%dcf = sqrt(abs(ktraj(:,:,1)/N));
%DCF_THRESHOLD = 0.25;
%mask = dcf > DCF_THRESHOLD; dcf = dcf.*mask + DCF_THRESHOLD*mask;
[Atb,Q] = giveOperators(kdata,ktraj,N,1);


% ==============================================================
% Compute the coil sensitivity map
% ============================================================== 

[coilImages] = coil_sens_map(Atb,Q,useGPU);
%csm=giveEspiritMaps(coilimages, nCh]));
imagesc(abs(coilImages(:,:,1)));
disp('Choose image region')
rect = round(getrect);
coilImages_Roi = coilImages(rect(2):rect(2)+rect(4),rect(1):rect(1)+rect(3),1:nChannels);

%% ===============================================================
% Compute coil compresession
% ================================================================
[vAtb,vCoilImages] = combine_coils_Atb(Atb,coilImages,0.95,coilImages);
nChannelsToChoose=size(vCoilImages,3);
csm=giveEspiritMaps(vCoilImages);
 %% coilImages=vcoilImages;
 clear vcoilImages vkdata;

% % ==============================================================
% % Compute the weight matrix
% % ============================================================== 


%% Trying static SENSE recon
%csm = ones(256);
testAtb = (squeeze(sum(vAtb,3)));
testAtb = sum(testAtb.*(conj(csm)),3);
testQ = sum(Q,3);
ATA = @(x) SenseATA(x,(testQ),(csm),N,1,nChannelsToChoose);
tic; testrecon =  pcg(ATA,(testAtb(:)),1e-5,150);toc
testrecon = gather(reshape(testrecon,[N,N]));

%%
lambda = 1;
if(useGPU)
    ATA = @(x) SenseATA_GPU(x,gpuArray(Q),gpuArray(csm),N,numFramesToKeep,nChannelsToChoose);
else
    ATA = @(x) SenseATA(x,(Q),(csm),N,numFramesToKeep,nChannelsToChoose);
end
    
%L = toeplitz([1,zeros(1,numFramesToKeep-1)],[1,-1,zeros(1,numFramesToKeep-2)]);
L = circulant([1,-1,zeros(1,numFramesToKeep-2)]);
%L = L(1:end-1,:); 
Lcirc = L'*L/2;


P = inv(eye(size(Lcirc,2)) + lambda*Lcirc); % Preconditioner
if(useGPU)
    ATA_lambda = @(x) ATA(x) + lambda*XL(x,N,numFramesToKeep,gpuArray(Lcirc));
    ATA_pre = @(x) reshape(reshape(ATA_lambda(x),N*N,numFramesToKeep)*gpuArray(P),[N*N*numFramesToKeep,1]);
    vAtbnew = (squeeze(sum(bsxfun(@times,vAtb,reshape(conj(csm),[N,N,1,nChannelsToChoose])),4)));
    vAtbnew = gpuArray(reshape(vAtbnew,[N*N,numFramesToKeep])*P);
    init = vAtbnew(:);
else
   ATA_lambda = @(x) ATA(x) + lambda*XL(x,N,numFramesToKeep,(L));
   ATA_pre = @(x) reshape(reshape(ATA_lambda(x),N*N,numFramesToKeep)*P,[N*N*numFramesToKeep,1]);
   vAtbnew = (squeeze(sum(bsxfun(@times,vAtb,reshape(conj(csm),[N,N,1,nChannelsToChoose])),4)));
   vAtbnew = reshape(vAtbnew,[N*N,numFramesToKeep])*P;
   init = vAtbnew(:);
end

%lowResRecons = l2ReconGPU(kdata_com,ktraj_com,coilImages,0.001);toc
tic; lowResRecons =  pcg(ATA_pre,(vAtbnew(:)),1e-5,65,[],[],init(:));toc
lowResRecons = gather(reshape(lowResRecons,[N,N,numFramesToKeep]));
vAtb1 = vAtb/(max(abs(lowResRecons(:))));
lowResRecons = lowResRecons/(max(abs(lowResRecons(:))));

%%
nav = lowResRecons(rect(2):rect(2)+rect(4),rect(1):rect(1)+rect(3),:);
[n1,n2,~] = size(nav); nav = reshape(nav,n1*n2,numFramesToKeep);
[Lnew,maxvalue] = estimateLapKernelLR(nav, 5);
%[K, X, L] = estimateLR(nav, 1,0.2,1);
init = lowResRecons;

%%
alpha = 0.1;
L = alpha*Lnew + (1-alpha)*Lcirc;
P = inv(eye(size(L,2)) + lambda*L); % Preconditioner
if(useGPU)
    ATA_lambda = @(x) ATA(x) + lambda*XL(x,N,numFramesToKeep,gpuArray(L));
    ATA_pre = @(x) reshape(reshape(ATA_lambda(x),N*N,numFramesToKeep)*gpuArray(P),[N*N*numFramesToKeep,1]);
    vAtbnew = (squeeze(sum(bsxfun(@times,vAtb1,reshape(conj(csm),[N,N,1,nChannelsToChoose])),4)));
    vAtbnew = gpuArray(reshape(vAtbnew,[N*N,numFramesToKeep])*P);
   % init = vAtbnew(:);
else
   ATA_lambda = @(x) ATA(x) + lambda*XL(x,N,numFramesToKeep,(L));
   ATA_pre = @(x) reshape(reshape(ATA_lambda(x),N*N,numFramesToKeep)*P,[N*N*numFramesToKeep,1]);
   vAtbnew = (squeeze(sum(bsxfun(@times,vAtb1,reshape(conj(csm),[N,N,1,nChannelsToChoose])),4)));
   vAtbnew = reshape(vAtbnew,[N*N,numFramesToKeep])*P;
  % init = vAtbnew(:);
end

%lowResRecons = l2ReconGPU(kdata_com,ktraj_com,coilImages,0.001);toc
tic; full =  pcg(ATA_pre,(vAtbnew(:)),1e-6,20,[],[],init(:));toc
full = gather(reshape(full,[N,N,numFramesToKeep]));
%%
nav = full(rect(2):rect(2)+rect(4),rect(1):rect(1)+rect(3),:);
[n1,n2,~] = size(nav); nav = reshape(nav,n1*n2,numFramesToKeep);
[Lnew,maxvalue] = estimateLapKernelLR(nav, 5);
%[K, X, L] = estimateLR(nav, 1,0.2,1);
init = full;
%for i=1:700,q = [squeeze(lowResRecons(:,80:170,i)),squeeze(full(:,80:170,i))];imagesc(abs(q),[0,0.0045]);colormap(gray);pause(0.03);end
%%

% [N1,~,Nframes] = size(lowResRecons);
% 
% csm_lowRes = giveEspiritMapsSmall(coilImages,N1,N1);
% ktraj_com = reshape(ktraj_com/N1,[size(ktraj_com,1),size(ktraj_com,2),numFramesToKeep]);
% kdata_com = reshape(kdata_com, [size(kdata_com,1),size(ktraj_com,2),numFramesToKeep,nChannelsToChoose]);
% FT_LR= NUFFT(ktraj_com,0*ktraj_com+1,0,0,[N1,N1]);
% [~,~,L]= est_laplacian_rev(reshape(lowResRecons,[N1*N1,Nframes]),FT_LR,kdata_com,csm_lowRes,N1, 4.5, 0.01);
% [~,Sbasis,V]=svd(L);
% V=V(:,end-nBasis+1:end);
% Sbasis=Sbasis(end-nBasis+1:end,end-nBasis+1:end);
% lambdaSmoothness = 0.02;
% 
% clear kdata_com ktraj_com
% 
% ktraj_scaled=reshape(ktraj_scaled,[nFreqEncoding*ninterleavesPerFrame,numFramesToKeep]);
% kdata=reshape(kdata,[nFreqEncoding*ninterleavesPerFrame,numFramesToKeep,nChannelsToChoose]);
% tic; x = solveUV(ktraj_scaled,kdata,csm, V, N, 50,lambdaSmoothness*Sbasis,useGPU);toc
% y = reshape(reshape(x,[N*N,nBasis])*V',[N,N,numFramesToKeep]);

