clear all
cd '/Users/jcb/abdul_brain/code'
addpath('/Users/jcb/MRSI/matlab_mns');
addpath('/Users/jcb/MRSI/MNS-Aug19/read_MR_MR27_7T');
addpath(genpath('/Users/jcb/medimgviewer'));

addpath(genpath('./../../gpuNUFFT'));
addpath(genpath('./../../Fessler-nufft'));
addpath(genpath('/Users/jcb/nfft2/matlab/nfft'));
addpath('/Users/jcb/bm4d/');
addpath(genpath('/Users/jcb/bm4d/MRIDenoisingPackage_r01_pcode'));

%d = './../17Jan20/P49152_rfs2.7';
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

s = [];
for i=1:3
    q = ifft(squeeze(dd(:,:,i:period:end)),[],2);
    s = [s;squeeze(q(1,:,:))];
    s = [s;squeeze(q(2,:,:))];
end
[u,l,v] = svd(s,'econ');
%%
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
params = transformKspaceAndData(tform,params);  
tic;p = params.FT'*bsxfun(@times,params.dd',params.dcf);toc  

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
  nvCh = min(4,nCK);%min(find(s>0.5));
  V = zeros(nCh,nvCh);
  V(channels_to_keep,:) = v(:,1:nvCh);

  dd_v = gather(V'*reshape(gpuArray(dd),nCh,nsamples*nangles));  
  dd_v = reshape(dd_v,nvCh,nsamples,nangles);

  %% Actual reconstruction        
mtx_reco = 340;
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
params = transformKspaceAndData(tform,params);  
tic;p = params.FT'*bsxfun(@times,params.dd',params.dcf);toc

[cmap,mask] = giveCSM(p,4,12);
cmap = bsxfun(@times,cmap,mask);
p1 = sum(p.*conj(cmap),4);

q = abs(p1)./(0.5*max(p1(:)));
VolumeViewer3D(flipdim(flipdim(squeeze(abs(q)),3),2));
