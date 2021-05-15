clear all
cd '/Users/jcb/abdul_brain'
clear all
cd '/Users/jcb/abdul_brain/code'
addpath('/Users/jcb/MRSI/matlab_mns');
addpath('/Users/jcb/MRSI/MNS-Aug19/read_MR_MR27_7T');

addpath(genpath('./../../gpuNUFFT'));
addpath(genpath('./../../Fessler-nufft'));
addpath(genpath('/Users/jcb/nfft2/matlab/nfft'));
addpath('/Users/jcb/bm4d/');
addpath(genpath('/Users/jcb/bm4d/MRIDenoisingPackage_r01_pcode'));
addpath(genpath('/Users/jcb/medimgviewer'));

d = './../31Jan20/P68608_woMotion.7';
%d = './../17Jan20/20200113_141044_P32768.7'; 
%d = './17Jan20/P52224.7'
wf_name = './../31Jan20/radial3D_1H_fov200_mtx400_intlv63200_kdt8_gmax33_smax118_dur5p2_fsca.mat';
%wf_name = './../17Jan20/radial3D_1H_fov224_mtx448_intlv101460_kdt4_gmax17_smax118_dur1p6_coca.mat';

[dd,k,dcf,mtx_acq,nangles,fov,phi,theta]  = readradial(d,[],wf_name);

nsamples = length(dcf)/nangles;

nCh = size(dd,1);
dd = reshape(dd,nCh,nsamples,nangles);
k = reshape(k,nsamples,nangles,3);
dcf = reshape(dcf,nsamples,nangles);

load(wf_name);
indices = find((phi==0) & (theta==0));
period = indices(5)-indices(4);

ind = ones(size(dd,2),size(dd,3));
for i=1:period:nangles,
    dd(:,:,i:i+2)=0;
    ind(:,i:i+2)=0;
end

%% Initial reconstruction        
mtx_reco = 100;
load(wf_name);

clear params;
params.dcf = dcf;
params.dd = dd(:,:,:);
params.phi = phi;
params.theta = theta;
params.nS = size(ks,2);
params.ks = ks*1.2*mtx_acq/mtx_reco; 
params.indices = squeeze(find(sum(abs(params.ks).^2,3)<0.25));

params.ks = params.ks(:,params.indices,:);
params.mtx_reco = mtx_reco;

tform = affine3d(eye(4));
params = transformKspaceAndDataCPU(tform,params);  

dd_new = params.dd;
dcf_new = params.dcf;
k = params.k;

p = zeros(mtx_reco,mtx_reco,mtx_reco,nCh);
parfor i=1:nCh,
    p(:,:,:,i) = gridding3dsingle(k,dd_new(i,:),dcf_new,[1 1 1]*mtx_reco);
end
%%
[cmap,mask] = giveCSM(p,4,12);
channels_to_keep = bsxfun(@times,p,not(mask));
channels_to_keep = squeeze(sum(sum(sum(abs(channels_to_keep).^2))));
[~,channels_to_keep] = sort(channels_to_keep,'ascend');

nCK = 16;
channels_to_keep = channels_to_keep(1:nCK);


disp('Coil Compression');
%tmp = gpuArray(dd(:,1:100,:));
%tmp = reshape(tmp,nCh,100*nangles);
tmp = reshape(p(:,:,:,channels_to_keep),mtx_reco^3,nCK);
Rs = gather(real(tmp'*tmp));

  [v,s] = eig(Rs);
  s = diag(s);[s,i] = sort(s,'descend');
  v = v(:,i);
  s=s./sum(s);s = cumsum(s);
  nvCh = min(6,nCK);%min(find(s>0.5));
  V = zeros(nCh,nvCh);
  V(channels_to_keep,:) = v(:,1:nvCh);

  
  dd_v = gather(V'*reshape(gpuArray(dd),nCh,nsamples*nangles));  
  dd_v = reshape(dd_v,nvCh,nsamples,nangles);

%% RECONSTRUCTION
mtx_reco = 300;
load(wf_name);

clear params;
params.dcf = dcf;
params.dd = dd_v(:,:,:);
params.phi = phi;
params.theta = theta;
params.nS = size(ks,2);
params.ks = ks*1.2*mtx_acq/mtx_reco; 
params.indices = squeeze(find(sum(abs(params.ks).^2,3)<0.25));

params.ks = params.ks(:,params.indices,:);
params.mtx_reco = mtx_reco;

tform = affine3d(eye(4));
params = transformKspaceAndDataCPU(tform,params);  
init_recon = zeros(mtx_reco,mtx_reco,mtx_reco,nvCh);

tic;
k= params.k;
dd_new = params.dd;
dcf_new = params.dcf;
parfor i=1:nvCh,
    init_recon(:,:,:,i) = gridding3dsingle(k,dd_new(i,:),dcf_new,[1 1 1]*mtx_reco);
end
toc
%%
[cmap,mask] = giveCSM(init_recon,6,15);
cmap = bsxfun(@times,cmap,mask);
cmap = gather(cmap);


% kspace weighting
%------------------
disp('Computing kspace weights');
ind_new = reshape(ind,params.nS,nangles);
ind_new = reshape(ind_new(params.indices,:),[1,length(params.indices)*nangles]);

kspaceWts = gridding3dsingle(k,ind_new,ones(size(dcf_new)),[1 1 1]*mtx_reco);
kspaceWts = gpuArray(abs(fftn(fftshift(kspaceWts))));

init_recon = bsxfun(@times,gpuArray(init_recon),gpuArray(mask));
disp('Computing Atb');

Atb = gpuArray(zeros(mtx_reco,mtx_reco,mtx_reco));
cmap = gpuArray(cmap);
for i=1:nvCh,
    temp = ifftshift(ifftn(fftn(fftshift(init_recon(:,:,:,i))).*kspaceWts));
    Atb = Atb + temp.*conj(cmap(:,:,:,i));
end
Atb = gather(Atb);
init_recon = gather(init_recon);
kspaceWts = gather(kspaceWts);
cmap = gather(cmap);
disp('Ready for CG');
clear temp;
%

niter = 300;pdenoised = 0*Atb;lambda =0.0;p1 = pdenoised;

T = FourierWtOp(kspaceWts,cmap,true);
G = TVOp(size(kspaceWts),[1,1,1],true);

M = @(x) T*x + lambda*(G*x);
 
tic;p1 = pcg(M,Atb(:)+lambda*gpuArray(pdenoised(:)),1e-14,niter,[],[],gpuArray(pdenoised(:)));toc
p1 = gather(p1);
p = reshape(p1,mtx_reco,mtx_reco,mtx_reco);
disp('CG recovery done');
clear T;
    
figure(1);imagesc(abs(p(:,:,mtx_reco/2)),[0,35]);drawnow;
    
%%
