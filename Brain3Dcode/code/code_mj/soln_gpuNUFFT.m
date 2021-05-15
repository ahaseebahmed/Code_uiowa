clear all
close all

addpath('/Users/jcb/MRSI/MNS-Aug19/read_MR_MR27_7T');
addpath('/Users/jcb/MRSI/matlab_mns');
addpath(genpath('./../gpuNUFFT'));
addpath(genpath('./../Fessler-nufft'));
addpath(genpath('/Users/jcb/nfft2/matlab/nfft'));

load data;
load k;

nangles = 101460;
nsamples = length(dcf)/nangles;

nCh = size(dd,1);
dd = reshape(dd,nCh,nsamples,nangles);
k = reshape(k,nsamples,nangles,3);
dcf = reshape(dcf,nsamples,nangles);

ind = ones(size(dd,2),size(dd,3));
for i=1:1140:nangles,
    dd(:,:,i:i+2)=0;
    ind(:,i:i+2)=0;
end
%tic;b1 = gridding3dsingle(k1,dd(1,:),dcf,[1 1 1]*mtx_reco);toc
%% LOW RESOLUTION RECONSTRUCTION
mtx_reco = 120;
mtx_acq = 480;
k1 = k*1.2*mtx_acq/mtx_reco; 

if(mtx_reco>mtx_acq)
    nS = nsamples;
else
    nS = floor(nsamples*mtx_reco/mtx_acq);
end

k1_new = reshape(k1(1:nS,:,:),nS*nangles,3);
dd_new = reshape(dd(:,1:nS,:),[nCh,nS*nangles]);
dcf_new = reshape(dcf(1:nS,:),nS*nangles,1);
 
osf = 2; % oversampling: 1.5 1.25
wg = 3; % kernel width: 5 7
sw = 8; % parallel sectors' width: 12 16

FT = gpuNUFFT(transpose(k1_new),ones(size(dcf_new)),osf,wg,sw,[1 1 1]*mtx_reco,[],true);

tic;init_recon = FT'*bsxfun(@times,dd_new(:,:)',(dcf_new));toc  

%%
  mtx_reco = 120;
  disp('Coil Compression');
  temp=reshape(init_recon,mtx_reco^3,nCh);
  Rs = real(temp'*temp);

  [v,s] = eig(Rs);
  s = diag(s);[s,i] = sort(s,'descend');
  v = v(:,i);
  s=s./sum(s);s = cumsum(s);
  nvchannels = 5;%min(find(s>0.5));

  dd_v = v(:,1:nvchannels)'*dd_new;  
  dd_v = dd_v';
  b1_v = FT'*bsxfun(@times,dd_v,dcf_new);
  
  disp('Estimate coil sensitivities.');
  [recon,cmap]=adapt_array_3d(b1_v);
  
%%
clear cmapfull;
mtx_reco = 180;
mtx_acq = 480;
k1 = k*1.2*mtx_acq/mtx_reco; 

% Modifying the kspace
if(mtx_reco>mtx_acq)
    nS = nsamples;
else
    nS = floor(nsamples*mtx_reco/mtx_acq);
end

k1_new = reshape(k1(1:nS,:,:),nS*nangles,3);
dd_new = reshape(dd(:,1:nS,:),[nCh,nS*nangles]);
dcf_new = reshape(dcf(1:nS,:),nS*nangles,1);
dd_v = v(:,1:nvchannels)'*dd_new;  
dd_v = dd_v';
ind_new = reshape(ind(1:nS,:),nS*nangles,1);
 
disp('Interpolating sensitivities to larger size.');
cmapfull = complex(zeros(mtx_reco,mtx_reco,mtx_reco,nvchannels));
cabs = (zeros(mtx_reco,mtx_reco,mtx_reco,nvchannels));

for i=1:nvchannels
        cmapfull(:,:,:,i)=imresize3(squeeze(real(cmap(:,:,:,i))),mtx_reco*[1,1,1],'linear')+ 1i.*imresize3(squeeze(imag(cmap(:,:,:,i))),mtx_reco*[1,1,1],'linear');
        cabs(:,:,:,i)=imresize3(squeeze(abs(cmap(:,:,:,i))),mtx_reco*[1,1,1],'linear');
  end
cmapfull = cabs.*exp(1i*angle(cmapfull));
clear cabs;

disp('Generate NUFFT Operator without coil sensitivities');
osf = 2; % oversampling: 1.5 1.25
wg = 3; % kernel width: 5 7
sw = 8; % parallel sectors' width: 12 16
FT = gpuNUFFT(transpose(k1_new),ones(size(dcf_new)),osf,wg,sw,[1 1 1]*mtx_reco,[],true);

tic;Atb = FT'*dd_v;toc
%tic;Atb = FT'*bsxfun(@times,dd_v(:,:),(dcf_new));toc  

Atb = sum(Atb.*conj(cmapfull),4);


FT1 = gpuNUFFT(transpose(k1_new),ones(size(dcf_new)),osf,wg,sw,[1 1 1]*2*mtx_reco,[],true);
kspaceWeights = FT1'*ind_new(:);


% FT1 = gpuNUFFT(transpose(k1_new),ones(size(dcf_new)),osf,wg,sw,[1 1 1]*mtx_reco,[],true);
% kspaceWeights = FT1'*ind_new(:);
% kspaceHolder = zeros([1 1 1]*2*mtx_reco);
% indices = mtx_reco/2+1:3*mtx_reco/2;
% kspaceHolder(indices,indices,indices) = kspaceWeights;
% kspaceWeights = kspaceHolder;

kspaceWeights = fftshift(kspaceWeights);
kspaceWeights = fftn(kspaceWeights);

T = ToeplitzOp((kspaceWeights),(cmapfull),true);
%T = ToeplitzOp(kspaceWeights,ones(mtx_reco,mtx_reco,mtx_reco),true);
M = @(x) T*x;

q = Atb;
tic;p1 = pcg(M,gpuArray(q(:)),1e-14,300);toc
p = reshape(p1,mtx_reco,mtx_reco,mtx_reco);
