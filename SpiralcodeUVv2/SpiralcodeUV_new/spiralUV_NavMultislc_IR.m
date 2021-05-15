 %clear all

addpath(genpath('./../../Data/SpiralData_Virg/Phantom'));
addpath('./csm');
addpath('./utils');
addpath('./Utils');
addpath('./extras');
addpath('./nufft_toolbox_cpu');
addpath(genpath('./gpuNUFFT'));
addpath(genpath('./CUDA'));

%% Reconstruction parameters
spiralsToDelete=0;
framesToDelete=0;
ninterleavesPerFrame=4;
N = 340;
nChannelsToChoose=14;
numFramesToKeep = 1000;
useGPU = 'true';
SHRINK_FACTOR = 1.0;
nBasis = 30;
lambdaSmoothness = 0.008;
cRatioI=[2,3,4,7,8,9,10,15,16,17,22,27,28,29];
%cRatioI=[1,2,3,8,9,11,12,13,14,15,16,17,19,20,21,22,23,24,25,26,29,30];
%cRatioI=1:nChannelsToChoose;
sigma=[4.5];
lam=[0.001];
% num_spiral_IR=700;
% num_IR=9;
%%
% % ==============================================================
% % Load the data
% % ============================================================== 
 %load(strcat('Series27.mat'),'kdata','k','dcf');
% dr1='./../../Data/SpiralData_2Oct18_UIOWA/Series10/';
 dr1=['/Shared/lss_jcb/DMRI_code/NUFFT/IRVirginia/'];

 dr2=pwd;
 cd(dr1);
 load('Spiral_cine_3T_1leave_FB_062_IR_FA15_TBW5_6_fixedSpiral_post_full.mat','kdata','ktraj','dcf');
 cd(dr2);
 
 kdata=squeeze(kdata);
  ktraj=cell2mat(ktraj);

 
 %kdata = permute(kdata,[1,3,2,4]);
 %[npt,nsp,nch,nsl]=size(kdata);
% kdata=kdata(:,1:ninterleavesPerFrame*numFramesToKeep,:);
 %
%  tmp=cat(2,kdata(:,1:702,:,:),kdata(:,701:706,:,:),kdata(:,703:2100,:,:),kdata(:,2097:2102,:,:)...
%      ,kdata(:,2101:2796,:,:),kdata(:,2795:2800,:,:),kdata(:,2797:4194,:,:),kdata(:,4191:4196,:,:)...
%      ,kdata(:,4195:4890,:,:),kdata(:,4889:4894,:,:),kdata(:,4891:6288,:,:),kdata(:,6287:6292,:,:)...
%      ,kdata(:,6289:6384,:,:));
%   ktmp=cat(2,k(:,1:702),k(:,701:706),k(:,703:2100),k(:,2097:2102)...
%      ,k(:,2101:2796),k(:,2795:2800),k(:,2797:4194),k(:,4191:4196)...
%      ,k(:,4195:4890),k(:,4889:4894),k(:,4891:6288),k(:,6287:6292)...
%      ,k(:,6289:6384));
%  k=reshape(ktmp,2336,6,[]);
%  kdata=tmp;
%  clear tmp ktmp;
%% 
%  %frm_IR()=
%  for i=1:num_IR
%  k_IR(:,:,i)=k(:,(i)*num_spiral_IR+1:(i)*num_spiral_IR+ninterleavesPerFrame);
%  end
%  
%  indx=[1 (1:num_IR-1)*num_spiral_IR+3];
%  for i=1:num_IR
% %  kdata_tmp1(:,:,:,:,i)=[kdata(:,indx(i):i*num_spiral_IR,:,:) kdata(:,i*num_spiral_IR+1:i*num_spiral_IR+2,:,:) ...
% %      kdata(:,i*num_spiral_IR+1:i*num_spiral_IR+6,:,:)]; 
%  k_tmp1(:,:,i)=[k(:,indx(i):i*num_spiral_IR) k(:,i*num_spiral_IR+1:i*num_spiral_IR+2) k(:,i*num_spiral_IR+1:i*num_spiral_IR+6)]; 
%     
%  %kdata(:,i*num_spiral_IR+1:num_spiral_IR*(i+1),:,:)];
%  end
%  i=1;
%  ii=1;
%  jj=1;
%  while (jj<=1075)
%      if ((ii*6)>700*i)
%          k_tmp1(:,:,jj)=k_IR(:,:,i);
%          i=i+1;
%      else
%          k_tmp1(:,:,jj)=k(:,(ii-1)*6+1:ii*6);
%          ii=ii+1;
%      end
%  jj=jj+1;
%  end
%  
%  kdata_tmp1 = permute(kdata_tmp1,[1,3,4,2,5]);
%  kdata = permute(kdata,[1,3,4,2]);
% 
%  kdata_tmp1=reshape(kdata_tmp1,[npt,nch,nsl,(num_spiral_IR+2)*num_IR]);
%  k_tmp1=reshape(k_tmp1,[npt,(num_spiral_IR+2)*num_IR]);
% 
%  kdata1=cat(4,kdata_tmp1,kdata(:,:,:,num_spiral_IR*num_IR+1:end));
%  k1=cat(2,k_tmp1,k(:,num_spiral_IR*num_IR+1:end));
% 
%  clear kdata kdata_tmp1 k k_tmp1;
%  kdata=kata1;clear kdata1;
%  k=k1; clear k1;

 %  i=i+1;
%  kdata_tmp2=[kdata(:,(i-1)*num_spiral_IR+1:i*num_spiral_IR,:,:) kdata(:,i*num_spiral_IR+1:i*num_spiral_IR+2,:,:)... 
%      kdata(:,i*num_spiral_IR+1:end,:,:)];
% k=k(:,1:ninterleavesPerFrame*numFramesToKeep);
% dcf=dcf(:,1:ninterleavesPerFrame*numFramesToKeep);
%%
% %-----------UIOWA Data----------------
% nex=8;
% nsp=800;
% kdata=squeeze(kdata);
% [nFreqEncoding,numberSpirals,nch,nslc]=size(kdata);
% % for i=1:numberSpirals/ninterleavesPerFrame
% %     kdata_tmp(:,:,i,:,:)=kdata(:,(i-1)*ninterleavesPerFrame+1:i*ninterleavesPerFrame,:,:);
% %     
% % end
% % kdata=reshape(kdata,[npt,nsp,nex,nch]);
% numFrames=floor(numberSpirals/ninterleavesPerFrame);
% kdata_slc=kdata(:,1:numFrames*ninterleavesPerFrame,:,:);
% k=k(:,1:numFrames*ninterleavesPerFrame);
% w=1;%dcf(:,1:numFrames*ninterleavesPerFrame);
% %kdata=reshape(kdata,[npt,numFrames*ninterleavesPerFrame*nex,nch]);
% clear kdata;
%% Slice Selection.
% for sl=4:size(kdata_slc,4)
%    nChannelsToChoose=8;
%     kdata=kdata_slc(:,:,:,sl);
% kdata=permute(kdata,[1,3,2]);
% k=k(:,1:numFrames*ninterleavesPerFrame);dcf=dcf(:,1:numFrames*ninterleavesPerFrame);
% k=repmat(k,[1,nex]);dcf=repmat(dcf,[1,nex]);
% k=cat(2,k,k,k,k);dcf=cat(2,dcf,dcf,dcf,dcf); 
% k=cat(2,k,k,k,k);dcf=cat(2,dcf,dcf,dcf,dcf); 

%------------Uni of virg----------%
% kdata=kdataUse;
%  k=ktrajmUse;
%  dcf=dcfUse;
%  clear kdataUse ktrajmUse dcfUse;
 %%
 
 %-------------Preprocessing Data-------------%
 
 [nFreqEncoding,nCh,numberSpirals]=size(kdata);
 numFrames=floor((numberSpirals-spiralsToDelete)/ninterleavesPerFrame);
% kdata=kdata(:,cRatioI(1:nChannelsToChoose),spiralsToDelete+1:numberSpirals);
% k = k(:,spiralsToDelete+1:numberSpirals);
% w=dcf(:,spiralsToDelete+1:numberSpirals);
% kdata=kdata(:,:,1:numFrames*ninterleavesPerFrame);
% k=k(:,1:numFrames*ninterleavesPerFrame);
% w=w(:,1:numFrames*ninterleavesPerFrame);
% 
% clear k;
 %dcf=ones(nFreqEncoding,1);
 
% Reshaping to frames
%[nFreqEncoding,numberSpirals,~,~]=size(kdata);
%kdata = permute(kdata,[1,3,2]);
kdata=kdata(:,:,1:numFrames*ninterleavesPerFrame);
ktraj=ktraj(:,1:numFrames*ninterleavesPerFrame);
kdata = permute(kdata,[1,3,2]);
kdata = reshape(kdata,[nFreqEncoding,ninterleavesPerFrame,numFrames,nCh]);
ktraj = reshape(ktraj,[nFreqEncoding,ninterleavesPerFrame,numFrames]);
%w = reshape(w,[nFreqEncoding,ninterleavesPerFrame,numFrames]);

% Keeping only numFramesToKeep

kdata = kdata(:,:,framesToDelete+1:numFramesToKeep+framesToDelete,cRatioI(1:nChannelsToChoose));
ktraj = ktraj(:,:,framesToDelete+1:numFramesToKeep+framesToDelete);
%save data kdata ktraj dcf
%%

%ktraj_scaled =  SHRINK_FACTOR*ktraj*N;
ktraj_scaled =  SHRINK_FACTOR*ktraj/(2*max(abs(ktraj(:))));
%ktraj_scaled =  SHRINK_FACTOR*ktraj;


%% ==============================================================
% Compute the coil sensitivity map
% ============================================================== 
ktraj_scaled=reshape(ktraj_scaled,[nFreqEncoding,ninterleavesPerFrame,numFramesToKeep]);
kdata=reshape(kdata,[nFreqEncoding,ninterleavesPerFrame,numFramesToKeep,nChannelsToChoose]);
%[coilImages] = coil_sens_map_NUFFT(kdata,ktraj_scaled,N,useGPU);
[coilImages1] = coil_sens_map_NUFFT(kdata(:,:,1:numFramesToKeep/2,:),ktraj_scaled(:,:,1:numFramesToKeep/2),N,useGPU);
[coilImages2] = coil_sens_map_NUFFT(kdata(:,:,1+(numFramesToKeep/2):end,:),ktraj_scaled(:,:,1+(numFramesToKeep/2):end),N,useGPU);

%% ===============================================================
% Compute coil compresession
% ================================================================
kdata=reshape(kdata,[nFreqEncoding*ninterleavesPerFrame*numFramesToKeep,nChannelsToChoose]);
[vkdata,vcoilImages] = combine_coils_covar(kdata,coilImages1,coilImages2,0.85);
nChannelsToChoose=size(vcoilImages,3);
coilImages=vcoilImages;
kdata=reshape(vkdata,[nFreqEncoding,ninterleavesPerFrame,numFramesToKeep,nChannelsToChoose]);
csm=giveEspiritMaps(reshape(coilImages,[size(coilImages,1), size(coilImages,2), nChannelsToChoose]),0.0025);
%coilImages=vcoilImages;

ktraj_scaled=reshape(ktraj_scaled,[nFreqEncoding,ninterleavesPerFrame,numFramesToKeep]);
kdata=reshape(kdata,[nFreqEncoding,ninterleavesPerFrame,numFramesToKeep,nChannelsToChoose]);
%% Sense Recon
% [kdata_com,ktraj_com] = binSpirals(kdata,ktraj_scaled,nFreqEncoding,11,1);
% ktraj_com = reshape(ktraj_com,[size(ktraj_com,1),size(ktraj_com,2),numFramesToKeep]);
% kdata_com = reshape(kdata_com, [size(kdata_com,1),size(ktraj_com,2),numFramesToKeep,nChannelsToChoose]);
% FT_LR= NUFFT(ktraj_com,1,0,0,[N,N]);
% tic;lowResRecons = l2Recont_v4(kdata_com,FT_LR,csm,0.01,N);toc


%% ==============================================================
% % Compute the weight matrix
% % ============================================================= 
no_ch=nChannelsToChoose;
Nav=squeeze(permute((kdata(:,1,:,:)),[1,2,4,3]));
Nav=Nav/max(abs(Nav(:)));
Nav=reshape(Nav,[nFreqEncoding*no_ch,numFramesToKeep]);

 q= fft(Nav,[],2);
 q(:,numFramesToKeep/2-numFramesToKeep/4-100:numFramesToKeep/2+numFramesToKeep/4+100)=0;
 Nav = ifft(q,[],2);
%Nav=Nav(:,:,1:264);
% Nav=reshape(permute(Nav,[1,3,2]),[nFreqEncoding*2,floor(numFramesToKeep/2),nChannelsToChoose]);
% Nav=permute(Nav,[1,3,2]);
[K,X,L]=estimateLapKernelLR(reshape(Nav,[nFreqEncoding*no_ch,numFramesToKeep]),4.5,0.001);
%load('/Shared/lss_jcb/abdul/IRdata/L_IR.mat');
[~,Sbasis,V]=svd(L);
V=V(:,end-nBasis+1:end);
Sbasis=Sbasis(end-nBasis+1:end,end-nBasis+1:end);
%% ==============================================================
% % Final Reconstruction
% % ============================================================= 
factor=0;
ktraj_scaled=reshape(ktraj_scaled,[nFreqEncoding*ninterleavesPerFrame,numFramesToKeep]);
kdata=reshape(kdata,[nFreqEncoding*ninterleavesPerFrame,numFramesToKeep,nChannelsToChoose]);
dcf=repmat(dcf,[1,ninterleavesPerFrame,numFramesToKeep]);
dcf=reshape(dcf,[nFreqEncoding*ninterleavesPerFrame,numFramesToKeep]);

osf = 2; wg = 3; sw = 8;
ktraj_gpu = [real(ktraj_scaled(:,1)),imag(ktraj_scaled(:,1))]';
FT = gpuNUFFT(ktraj_gpu,dcf(:,1),osf,wg,sw,[N,N],ones(N,N),true);
t_imp=zeros(N,N);
t_imp(N/2,N/2)=1;
i_tmp=FT'*(FT*t_imp);
dcf1=dcf./i_tmp(N/2,N/2);

%tic; x = solveUV1(ktraj_scaled,dcf,kdata,csm, V, N, 10,lambdaSmoothness*Sbasis,useGPU,factor);toc
tic; x = solveUV(ktraj_scaled,dcf1,kdata,csm, V, N, 100,lambdaSmoothness*Sbasis,useGPU,factor);toc

% for i=1:1
%     factor=(abs(x)+mean(abs(x(:))).*10);
%     tic; x = solveUV(ktraj_scaled,kdata,csm, V, N, 30,lambdaSmoothness*Sbasis,useGPU,factor);toc
% end

y = reshape(reshape(x,[N*N,nBasis])*V',[N,N,numFramesToKeep]);

%% ==============================================================
% % Save and Display results
% % ============================================================= 

%for i=1:530;imagesc(fliplr(flipud(abs(y(:,:,i))))); pause(0.1); colormap gray;end
%for i=1:250;imagesc(rot90(flipud(abs(y(:,:,i))))); pause(0.1); colormap gray; end
%clear kdata csm V ;
%mkdir './../../Data/SpiralData_24Aug18_UIOWA/Series17/';
%cd './../../Data/SpiralData_Virg/Phantom/3T_';
cd(dr1);
save(strcat('Spiral_cine_3T_1leave_FB_062_IR_FA15_TBW5_6_fixedSpiral_post_full28',num2str(lambdaSmoothness),'_',num2str(sigma),'_',num2str(lam),'.mat'), 'y','L','-v7.3');
cd(dr2);
%cd './../../../SpiralcodeUVv2/SpiralcodeUV_new';

%end

% 
% for i=1:8;colormap gray;
%     subplot(2,4,i);imagesc(((abs(csm(:,:,i)))));
% end
% figure;hold on;
%  for i=1:30
%      subplot(6,5,i);
%      plot(V(:,end+1-i));
%  end
%50:270,106:230 IR data