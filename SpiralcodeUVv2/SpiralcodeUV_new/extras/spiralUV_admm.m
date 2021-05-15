clear all

addpath('./../../Data');
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
nChannelsToChoose=4;
numFramesToKeep = 580;
%% Reconstruction parameters
maxitCG = 15;
alpha = 1e-5;
tol = 1e-6;
display = 0;
imwidth = N;
useGPU = 'true';
SHRINK_FACTOR = 1.3;
nBasis = 20;
lambdaSmoothness = 0.01;

%%
% % %
% % ==============================================================
% % Load the data
% % ============================================================== 
%Load data
 load('./../../Data/Spiral_cine_3T_1leave_FB_027_full.mat','kdata','ktraj','dcf');
 load('./../../Data/cRatioI_027.mat');
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
[csm,coilImages] = coil_sens_map_NUFFT(kdata,ktraj_scaled,N,useGPU);

% % ==============================================================
% % Compute the weight matrix
% % ============================================================== 
%%
ktraj_scaled=reshape(ktraj_scaled,[nFreqEncoding,ninterleavesPerFrame,numFramesToKeep]);
kdata=reshape(kdata,[nFreqEncoding,ninterleavesPerFrame,numFramesToKeep,nChannelsToChoose]);

[kdata_com,ktraj_com] = binSpirals(kdata,ktraj_scaled,1600,5);
N1 = 64;%ceil(max([max(abs(real(ktraj_com(:)))),max(abs(imag(ktraj_com(:))))]));
csm_lowRes = giveEspiritMapsSmall(coilImages,N1,N1);
ktraj_com = reshape(ktraj_com/N1,[size(ktraj_com,1),size(ktraj_com,2),numFramesToKeep]);
kdata_com = reshape(kdata_com, [size(kdata_com,1),size(ktraj_com,2),numFramesToKeep,nChannelsToChoose]);
FT_LR= NUFFT(ktraj_com,1,0,0,[N1,N1]);
tic;lowResRecons = l2Recont_v2(kdata_com,FT_LR,csm_lowRes,0.01);toc
ktraj_com=reshape(ktraj_com,[size(ktraj_com,1)*size(ktraj_com,2),numFramesToKeep]);
kdata_com=reshape(kdata_com,[size(kdata_com,1)*size(kdata_com,2),numFramesToKeep,nChannelsToChoose]);
osf = 2; wg = 3; sw = 8;

ktraj_gpu = [real(ktraj_com(:)),imag(ktraj_com(:))]';
FT = gpuNUFFT(ktraj_gpu,ones(size(ktraj_com,1)*size(ktraj_com,2),1),osf,wg,sw,[N1,N1],[],true);

tic
for ii=1:10
[~,Si,L]=estimateLapKernelLRv2(reshape(Recons,[N*N,numFramesToKeep]),reshape(Si1,[N1*N1,numFramesToKeep]),2.5,0.025);
%[~,Si,L]=LRrecon(reshape(lowResRecons,[N1*N1,numFramesToKeep]),2.5,1);
[~,Si1,L1]=estimateLapKernelLR(reshape(lowResRecons,[N1*N1,numFramesToKeep]),2.5,1);


[~,Sbasis,V]=svd(L);
V=V(:,end-nBasis+1:end);
Sbasis=Sbasis(end-nBasis+1:end,end-nBasis+1:end);

Atb = Atb_UV(FT,kdata_com,V,csm_lowRes,true);
Reg = @(x) reshape(reshape(x,[N1*N1,nBasis])*Sbasis,[N1*N1*nBasis,1]);
AtA = @(x) AtA_UV(FT,x,V,csm_lowRes,size(kdata_com,1)) + Reg(x);
tic; x1 = pcg(AtA,Atb(:),1e-5,50);toc;
lowResRecons1 = reshape(reshape(x1,[N1*N1,nBasis])*V',[N1,N1,numFramesToKeep]);
end
toc;


%%
% ktraj_scaled=reshape(ktraj_scaled,[nFreqEncoding*ninterleavesPerFrame,numFramesToKeep]);
% kdata=reshape(kdata,[nFreqEncoding*ninterleavesPerFrame,numFramesToKeep,nChannelsToChoose]);
beta=0.1;
q=0.5;
gamma = 100;
eta=2;
sigSq=2.5;
Y=zeros(N*N,nBasis);
Z=zeros(N*N,nBasis);
U=zeros(N*N,nBasis);


Sb=lambdaSmoothness*diag(Sbasis)+beta;
Sb=diag(Sb);

while(1)


[~,Sbasis,V]=svd(L);
V=V(:,end-nBasis+1:end);
Sbasis=Sbasis(end-nBasis+1:end,end-nBasis+1:end);

Zpre=Z;
U = solveUV_admm(ktraj_scaled,kdata,csm, V, N, Z,Y,dcf,useGPU);
Z = (U+Y/beta)*Sb;
Y = Y+ beta*(U-Z);

if norm(Zpre-Z)<1e-4 && norm(U-Z)< 1e-4
    break;
end

    
    gamma = gamma/eta;
    
    X2 = sum(Z.*conj(Z),1);
    X3 = (Z')*Z;
    dsq = abs(repmat(X2,nf,1)+repmat(X2',1,nf)-2*real(X3));
    K = exp(-dsq/sigSq);
    
    [V,S,~] = svd(K);
    W = V*((S+gamma*eye(nf))^(-q))*V';
    A = W.*K;
    L = -diag(sum(A))+A;

end


%%

%clear kdata_com ktraj_com

ktraj_scaled=reshape(ktraj_scaled,[nFreqEncoding*ninterleavesPerFrame,numFramesToKeep]);
kdata=reshape(kdata,[nFreqEncoding*ninterleavesPerFrame,numFramesToKeep,nChannelsToChoose]);
tic; x = solveUV(ktraj_scaled,kdata,csm, V, N, 50,lambdaSmoothness*Sbasis,dcf,useGPU);toc
y = reshape(reshape(x,[N*N,nBasis])*V',[N,N,numFramesToKeep]);
save('res_wo_storm1.mat','y');
