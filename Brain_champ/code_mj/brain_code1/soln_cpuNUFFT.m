cd '/Users/jcb/abdul_brain'
addpath('/Users/jcb/MRSI/MNS-Aug19/read_MR_MR27_7T');
addpath('/Users/jcb/MRSI/matlab_mns');
addpath(genpath('./../gpuNUFFT'));
addpath(genpath('./../Fessler-nufft'));
addpath(genpath('/Users/jcb/nfft2/matlab/nfft'));
addpath('/Users/jcb/bm4d/');
addpath(genpath('/Users/jcb/bm4d/MRIDenoisingPackage_r01_pcode'));

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

%%
disp('Coil Compression');
tmp = gpuArray(dd(:,1:100,:));
tmp = reshape(tmp,nCh,100*nangles);
Rs = gather(real(tmp*tmp'));

  [v,s] = eig(Rs);
  s = diag(s);[s,i] = sort(s,'descend');
  v = v(:,i);
  s=s./sum(s);s = cumsum(s);
  nvCh = 5;%min(find(s>0.5));

  dd_v = gather(v(:,1:nvCh)'*reshape(gpuArray(dd),nCh,nsamples*nangles));  
  dd_v = reshape(dd_v,nvCh,nsamples,nangles);
%tic;b1 = gridding3dsingle(k1,dd(1,:),dcf,[1 1 1]*mtx_reco);toc
%% LOW RESOLUTION RECONSTRUCTION
mtx_reco = 420;
mtx_acq = 480;
k1 = k*1.1*mtx_acq/mtx_reco; 

if(mtx_reco>mtx_acq)
    nS = nsamples;
else
    nS = floor(nsamples*mtx_reco/mtx_acq);
end

k1_new = reshape(k1(1:nS,:,:),nS*nangles,3);
dd_new = reshape(dd_v(:,1:nS,:),[nvCh,nS*nangles]);
dcf_new = reshape(dcf(1:nS,:),nS*nangles,1);
ind_new = reshape(ind(1:nS,:),nS*nangles,1);

init_recon = zeros(mtx_reco,mtx_reco,mtx_reco,nvCh);

tic;
parfor i=1:nvCh,
init_recon(:,:,:,i) = gridding3dsingle(k1_new,dd_new(i,:),dcf_new,[1 1 1]*mtx_reco);
end
toc

[cmap,mask] = giveCSM(init_recon,4,12);
cmap = bsxfun(@times,cmap,mask);
cmap = gather(cmap);

% Heurestic; scaling the cmaps by sos of cmaps to normalize
sos = sqrt(sum(abs(cmap).^2,4));
sos = sos+not(mask);
cmapnew = bsxfun(@times,cmap,1./sos);

% kspace weighting
%------------------
disp('Computing kspace weights');

kspaceWts = gridding3dsingle(k1_new,ind_new(:)',ones(size(dcf_new)),[1 1 1]*mtx_reco);
kspaceWts = gpuArray(abs(fftn(fftshift(kspaceWts))));

init_recon = bsxfun(@times,gpuArray(init_recon),gpuArray(mask));
disp('Computing Atb');

Atb = gpuArray(zeros(mtx_reco,mtx_reco,mtx_reco));
cmap = gpuArray(cmap);
for i=1:nvCh,
    temp = ifftshift(ifftn(fftn(fftshift(init_recon(:,:,:,i))).*kspaceWts));
    Atb = Atb + temp.*conj(cmapnew(:,:,:,i));
end
Atb = gather(Atb);
init_recon = gather(init_recon);
kspaceWts = gather(kspaceWts);
cmap = gather(cmap);
disp('Ready for CG');
clear temp;
%
%%
niter = 300;pdenoised = 0*Atb;lambda =.2;p1 = pdenoised;

T = FourierWtOp(kspaceWts,cmapnew,true);
G = TVOp(size(kspaceWts),[1,1,1],true);

M = @(x) T*x + lambda*(G*x);
 
tic;p1 = pcg(M,Atb(:)+lambda*gpuArray(pdenoised(:)),1e-14,niter,[],[],gpuArray(pdenoised(:)));toc
p1 = gather(p1);
p = reshape(p1,mtx_reco,mtx_reco,mtx_reco);
disp('CG recovery done');
clear T;
    
figure(1);imagesc(abs(p(:,:,mtx_reco/2)),[0,300]);drawnow;
    
%%
