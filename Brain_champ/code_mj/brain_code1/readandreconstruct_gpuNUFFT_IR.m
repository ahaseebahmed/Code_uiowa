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

d = '/Users/jcb/abdul_brain/scan17FebHuman/P88576.7'
wf_name = '/Users/jcb/abdul_brain/scan17FebHuman/radial3D_1H_fov200_mtx800_intlv206550_kdt6_gmax16_smax119_dur3p2_coca'


%d='/Users/jcb/abdul_brain/31Jan20/P97792_0.75Res_IR.7';
%wf_name = '/Users/jcb/abdul_brain/31Jan20/radial3D_1H_fov192_mtx256_intlv206625_kdt4_gmax24_smax120_dur1_coca.mat';

%d='/Users/jcb/abdul_brain/31Jan20/highRes/P53248_1ResHuman.7';
%wf_name = '/Users/jcb/abdul_brain/31Jan20/highRes/radial3D_1H_fov200_mtx200_intlv126300_kdt4_gmax23_smax120_dur0p8_coca.mat';


[dd,k,dcf,mtx_acq,nangles,fov,phi,theta]  = readradial(d,[],wf_name,[],[],[],[],...
    [],[],true,true);

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
    dcf(:,i:i+2)=0;
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
tmp = coilImages(:,:,:,:);
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
mtx_reco = 300;
nWraps = 15;
nangles = size(dd_v,3);
nAcqsInWrap = nangles/nWraps;
nSpokesinIR=375;
n_IRs = 400;%floor(nangles/nSpokesinIR);

nudgefactor = 1.2;
mtx_reco = mtx_reco+mod(mtx_reco,2);
k1= k*nudgefactor*mtx_acq/mtx_reco;

kmag = max(sqrt(sum(abs(k1).^2,3)),[],2);
indices = kmag<0.5;
k1 = k1(indices,:,:);
dcf1 = dcf(indices,:);
dd_v1 = dd_v(:,indices,:);

nsamplesNew = size(k1,1);

clear dd_2
clear k2
clear dcf2;
%dd_v1 = zeros(nvCh,nsamples,nSpokesinIR,210);
startSpoke = 1;
for nIR=1:400%floor(nangles/nSpokesinIR) 
    dd_2(:,:,:,nIR) = reshape(dd_v(:,:,startSpoke:startSpoke+nSpokesinIR-1),[nvCh,nsamples,nSpokesinIR,1]);
    k2(:,:,nIR,:) = k1(:,startSpoke:startSpoke+nSpokesinIR-1,:);
    dcf2(:,:,nIR) = dcf1(:,startSpoke:startSpoke+nSpokesinIR-1);

    startSpoke = startSpoke+nSpokesinIR;
    
    if(mod(startSpoke+nSpokesinIR,nAcqsInWrap)<=nSpokesinIR)
        startSpoke = startSpoke+2*nSpokesinIR-mod(startSpoke+nSpokesinIR,nAcqsInWrap);
    end
    if(startSpoke>nangles)
        break
    end
end

dd_2 = dd_2(:,indices,:,2:end);
nIRReduced = size(dd_2,4);
k2 = k2(indices,:,2:end,:);
dcf2 = dcf2(indices,:,2:end);

%% Computation of coil sensitivities
sz = size(k2);
ktemp = reshape(k2,prod(sz(1:3)),3);
osf = 2; wg = 3; sw = 8; 

FT = gpuNUFFT(transpose(ktemp),ones(prod(sz(1:3)),1),osf,wg,sw,[1 1 1]*mtx_reco,[],true);
dtemp = reshape(dd_2,[nvCh,prod(sz(1:3))]);
dcftemp = dcf2; 
dcftemp = reshape(dcftemp,1,prod(sz(1:3)));
tic;coilImages = FT'*bsxfun(@times,dtemp,dcftemp)';toc
sos = sqrt(sum(abs(coilImages).^2,4));
imagesc(imrotate(squeeze(abs(sos(floor(mtx_reco/2),:,:))),-90),[0,2e6])
    
[cmap,mask] = giveCSM(coilImages,5,12,0.01);%
cmap = bsxfun(@times,cmap,mask);

SOS = sum(coilImages.*conj(cmap),4);
figure(1);imagesc((abs(SOS(:,:,floor(mtx_reco/2)))),[0,0.25*max(abs(SOS(:)))]);

%% Computation of Atb for each bin

%indicesStart = [2,25,51,101,201];
indicesStart = [55,65,75,85,100];

numBins = length(indicesStart)
indicesEnd = indicesStart + [25,25,25,25,200];
%indicesEnd = indicesStart + [50,50,50,50,249];0

Atb = zeros(mtx_reco,mtx_reco,mtx_reco,nvCh,numBins);
kspaceWts = zeros(2*mtx_reco,2*mtx_reco,2*mtx_reco,numBins);
sos = zeros(mtx_reco,mtx_reco,mtx_reco,numBins);

for i=1:length(indicesStart),
    i
    
    spokesInBin = indicesStart(i):indicesEnd(i);    
    nkBin = nsamplesNew*length(spokesInBin)*nIRReduced;
    kbin = k2(:,indicesStart(i):indicesEnd(i),:,:);
    kbin = reshape(kbin,nkBin,3);
    dcfbin = dcf2(:,indicesStart(i):indicesEnd(i),:); %dcftemp(:,1:3,:)=0;%dcftemp(:,[450:end],:)=0;
    dcfbin = reshape(dcfbin,1,nkBin);
    
    osf = 2; wg = 3; sw = 8; % parallel sectors' width: 12 16
    FT = gpuNUFFT(transpose(kbin),ones(size(dcfbin)),osf,wg,sw,[1 1 1]*mtx_reco,[],true);
    
    dd_bin = dd_2(:,:,indicesStart(i):indicesEnd(i),:);
    dd_bin = reshape(dd_bin,[nvCh,nkBin]); 
        
    Atb(:,:,:,:,i) = FT'*bsxfun(@times,dd_bin,dcfbin)'.*50./length(spokesInBin);
    sos(:,:,:,i) = sqrt(sum(abs(Atb(:,:,:,:,i)).^2,4));    
    kspaceWts(:,:,:,i) = giveNUFFTx2(dcfbin(:),FT,mtx_reco,kbin');
    kspaceWts(:,:,:,i) = (fftn(fftshift(kspaceWts(:,:,:,i))));

    imagesc(imrotate(squeeze(abs(sos(floor(mtx_reco/2),:,:,i))),-90)); drawnow
end

%%
clear FT
T = FourierWtOpSeries(gpuArray(kspaceWts),gpuArray(cmap));
G = GradOp([mtx_reco,mtx_reco,mtx_reco,numBins],[2,2,2,1]);

M = @(x) T*x+1e-4*(G*x);

Atb1 = sum(bsxfun(@times,Atb,conj(cmap)),4);
tic;recon = pcg(M,gpuArray(Atb1(:)),1e-14,100,[],[],gpuArray(Atb1(:)));toc

recon = gather(recon);
recon = reshape(recon,mtx_reco,mtx_reco,mtx_reco,numBins);
disp('CG recovery done');
clear T;

sca =[0.2,0.1,0.1,0.1,0.2];
for i=1:size(recon,4)
    figure(i);tmp = squeeze(abs(recon(floor(mtx_reco/2),:,:,i)));
    imagesc(imrotate(squeeze(tmp),-90),[0,0.4*max(tmp(:))]); colormap(gray);drawnow
    axis('image'); axis off
end

for i=1:size(recon,4)
    figure(i+10);tmp = squeeze(abs(Atb1(floor(mtx_reco/2),:,:,i)));
    imagesc(imrotate(squeeze(tmp),-90),[0,0.4*max(tmp(:))]); colormap(gray);drawnow
    axis('image'); axis off
end

% sca =[0.2,0.2,0.2,0.2,0.2];
% for i=1:size(recon,4)
%     figure(i+10);tmp = squeeze(abs(recon(:,74,:,i)));
%     imagesc(imrotate(squeeze(tmp),-90),[0,0.4*max(tmp(:))]); drawnow
%     colormap(gray);axis('image'); axis off
% end
