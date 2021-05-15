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

d='/Users/jcb/abdul_brain/31Jan20/P97792_0.75Res_IR.7';
wf_name = '/Users/jcb/abdul_brain/31Jan20/radial3D_1H_fov192_mtx256_intlv206625_kdt4_gmax24_smax120_dur1_coca.mat';

%d='/Users/jcb/abdul_brain/31Jan20/highRes/P53248_1ResHuman.7';
%wf_name = '/Users/jcb/abdul_brain/31Jan20/highRes/radial3D_1H_fov200_mtx200_intlv126300_kdt4_gmax23_smax120_dur0p8_coca.mat';


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

ind = ones(size(dd,2),size(dd,3));
for i=1:period:nangles,
    dd(:,:,i:i+2)=0;
    ind(:,i:i+2)=0;
end

%% Initial reconstruction at lower resolution
mtx_reco = 100;
load(wf_name);

clear params;
params.dcf = dcf;
params.dd = dd;%dd_v(:,:,:);
params.phi = phi;
params.theta = theta;
params.nS = size(ks,2);
params.ks = ks*1.2*mtx_acq/mtx_reco; 
params.indices = squeeze(find(sum(abs(params.ks).^2,3)<0.25));

params.ks = params.ks(:,params.indices,:);
params.mtx_reco = mtx_reco;

tform = affine3d(eye(4));
params = transformKspaceAndData(tform,params);  
tic;coilImages = params.FT'*bsxfun(@times,params.dd',(params.dcf(:)));toc  
sos = sqrt(sum(abs(coilImages).^2,4));

mask = abs(sos)<0.22*max(sos(:));
coilImages = bsxfun(@times,coilImages,mask);
%[denoised_norm,biasfield] = giveNormalizedImage(abs(test1),.15,1);

disp('Coil Compression');
tmp = coilImages(:,:,50:end,:);
[n1,n2,n3,~] = size(tmp);
tmp = reshape(tmp,n1*n2*n3,nCh);
Rs = (real(tmp'*tmp));

  [v,s] = eig(Rs);
  s = diag(s);[s,i] = sort(s,'descend');
  v = v(:,i);
  s=s./sum(s);s = cumsum(s);
  nvCh = min(5,nCh);%min(find(s>0.9));

  dd_v = (v(:,1:nvCh)'*reshape((dd),nCh,nsamples*nangles));  
  dd_v = reshape(dd_v,nvCh,nsamples,nangles);

%%
mtx_reco=340;
nWraps = 10;
nangles = size(dd_v,3);
nAcqsInWrap = nangles/nWraps;
nSpokesinIR=600;
n_IRs = floor(nangles/nSpokesinIR);

clear dd_1
clear k1
clear dcf1;
%dd_v1 = zeros(nvCh,nsamples,nSpokesinIR,210);
startSpoke = 1;
for nIR=1:floor(nangles/nSpokesinIR) 
    dd_1(:,:,:,nIR) = reshape(dd(:,:,startSpoke:startSpoke+nSpokesinIR-1),[nCh,151,600,1]);
    k1(:,:,nIR,:) = k(:,startSpoke:startSpoke+nSpokesinIR-1,:);
    dcf1(:,:,nIR) = dcf(:,startSpoke:startSpoke+nSpokesinIR-1);

    startSpoke = startSpoke+nSpokesinIR;
    
    if(mod(startSpoke+nSpokesinIR,nAcqsInWrap)<=nSpokesinIR)
        startSpoke = startSpoke+2*nSpokesinIR-mod(startSpoke+nSpokesinIR,nAcqsInWrap);
    end
    if(startSpoke>nangles)
        break
    end
end

dd_1 = dd_1(:,:,:,2:end);
nIRReduced = size(dd_1,4);
k1 = k1(:,:,2:end,:);
dcf1 = dcf1(:,:,2:end);

%% Final recon at 340 matrix size
mtx_reco = 340

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
tic;coilImages = params.FT'*bsxfun(@times,params.dd',(params.dcf(:)));toc  
sos = sqrt(sum(abs(coilImages).^2,4));
[cmap,mask] = giveCSM(coilImages,5,14,0.07);
cmap = bsxfun(@times,cmap,mask);
Atb = sum(coilImages.*conj(cmap),4);

%%
indnew = ind(params.indices,:); 
indnew(:,1:period:end)=0;
indnew(:,2:period:end)=0;
indnew(:,3:period:end)=0;
indnew = indnew(:).*params.dcf(:);

kspaceWts = giveNUFFTx2(indnew(:),params.FT,params.mtx_reco,params.k');
kspaceWts = (fftn(fftshift(kspaceWts)));

clear A
A = FourierWtOpSeries(,ones(mtx_reco*[1,1,1]),true);
G = GradOp(mtx_reco*[1,1,1],[1,1,1]);

M = @(x) A*x + 2e3*(G*x);
%M = @(x) A*x + 5e-10*x;
%M = @(x) reshape(params.FT'*(params.FT*reshape(x,mtx_reco*[1,1,1])),[mtx_reco^3,1]);
smoothedImages = zeros(size(coilImages));
for i=1:nvCh
    Atb = coilImages(:,:,:,i);
    tic;temp = pcg(M,gpuArray(Atb(:)),1e-4,10);toc
    temp = gather(reshape(temp,mtx_reco*[1,1,1]));
    smoothedImages(:,:,:,i) = temp;
end
sos = sqrt(sum(abs(smoothedImages).^2,4));
imagesc(abs(sos(:,:,151)),[0,300]);


%% testing giveNUFFTx2

% ktest = (params.k)';
% 
% mtx_reco = 100;
% FT1 = gpuNUFFT((ktest),ones(size(ktest,2),1),osf,wg,sw,2*mtx_reco*[1,1,1],[],true);
% FT2 = gpuNUFFT((ktest),ones(size(ktest,2),1),osf,wg,sw,mtx_reco*[1,1,1],[],true);
% 
% q1 = (FT1'*indnew(:));


% n=101;
% figure(1); imagesc(abs(q1(:,:,n)));
% figure(2); imagesc(abs(q3(:,:,n)));

%q2 = q2.*(q1(101)/q2(101));
%%
% for i=0.5:-1:-0.5,
%     for j=0.5:-1:-0.5,
%         for k=0.5:-1:-0.5,
%             indshifed(:,n) = indnew(:).*exp(1i*2*pi*(params.k*mtx_reco*[i;j;k]));
%             n=n+1;
%         end
%     end
% end
kspaceWts = zeros(mtx_reco*2*[1,1,1]);
for i=-0.5:1:0.5,
    for j=-0.5:1:0.5,
        for k=-0.5:1:0.5,
            indshifed(:,n) = indnew(:).*exp(1i*2*pi*(params.k*mtx_reco*[i;j;k]));
            n=n+1;
        end
    end
end
%kspaceWts = params.FT'*ones(size(params.dd,2),1);
%Wts = zeros(2*mtx_reco*[1,1,1]);
%indices = [mtx_reco/2+1:2*mtx_reco-mtx_reco/2]';
%Wts(indices,indices,indices)=kspaceWts;
Wts = (fftn(fftshift(Wts)));
A = FourierWtOpSeries(Wts,ones(mtx_reco*[1,1,1]),true);
%%
M = @(x) A*x;
%M = @(x) reshape(params.FT'*(params.FT*reshape(x,mtx_reco*[1,1,1])),[mtx_reco^3,1]);
temp = pcg(M,gpuArray(Atb(:)),1e-4,300);
temp = reshape(temp,[100,100,100]);
imagesc(abs(temp(:,:,51)));
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
  nvCh = min(4,nCK);%min(find(s>0.5));
  V = zeros(nCh,nvCh);
  V(channels_to_keep,:) = v(:,1:nvCh);

  dd_v = gather(V'*reshape(gpuArray(dd),nCh,nsamples*nangles));  
  dd_v = reshape(dd_v,nvCh,nsamples,nangles);

%%
  
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
%%


%%
