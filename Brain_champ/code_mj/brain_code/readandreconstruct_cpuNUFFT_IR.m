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

%d='/Users/jcb/abdul_brain/31Jan20/P97792_0.75Res_IR.7';
%wf_name = '/Users/jcb/abdul_brain/31Jan20/radial3D_1H_fov192_mtx256_intlv206625_kdt4_gmax24_smax120_dur1_coca.mat';
d = '/Users/jcb/abdul_brain/scan17FebHuman/P88576.7'
wf_name = '/Users/jcb/abdul_brain/scan17FebHuman/radial3D_1H_fov200_mtx800_intlv206550_kdt6_gmax16_smax119_dur3p2_coca'


[dd,k,dcf,mtx_acq,nangles,fov,phi,theta]  = readradial(d,[],wf_name);

nsamples = length(dcf)/nangles;
%c_ind=[2:7,10:15,18,19,22,23,26:27,30,31];
%dd=dd(c_ind,:);
nCh = size(dd,1);
dd = reshape(dd,nCh,nsamples,nangles);
k = reshape(k,nsamples,nangles,3);
dcf = reshape(dcf,nsamples,nangles);
ind = ones(size(dd,2),size(dd,3));


indices = find((phi==0) & (theta==0));
period = indices(5)-indices(4);

dcf(:,1:period:end)=0;
dcf(:,2:period:end)=0;
dcf(:,3:period:end)=0;

%% Initial reconstruction

nudgefactor = 1.1;
mtx_reco = mtx_acq;
mtx_reco = mtx_reco+mod(mtx_reco,2);

osf = 2; wg = 3; sw = 8; 
k1= k*nudgefactor*mtx_acq/mtx_reco;
nangles = size(dd,3);

kmag = max(sqrt(sum(abs(k1).^2,3)),[],2);
indices = kmag<0.5;
k1 = k1(indices,:,:);
nsamplesNew = size(k1,1);

sz = size(k1);
ktemp = reshape(k1,prod(sz(1:2)),3);

FT = gpuNUFFT(transpose(ktemp),ones(prod(sz(1:2)),1),osf,wg,sw,[1 1 1]*mtx_reco,[],true);
dtemp = reshape(dd(:,indices,:),[nCh,nsamplesNew*nangles]);
dcftemp = reshape(dcf(indices,:),[1,nsamplesNew*nangles]);
tic;coilImages = FT'*bsxfun(@times,dtemp,dcftemp)';toc
sos = sqrt(sum(abs(coilImages).^2,4));
imagesc(imrotate(squeeze(abs(sos(floor(mtx_reco/2),:,:))),-90))

%%
disp('Coil Compression');

% use this step to exclude the dielectric pad
tmp = coilImages(30:end,:,:,:);


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
  
%% Scanner based sorting
nudgefactor = 1.2;
mtx_reco = 300;
mtx_reco = mtx_reco+mod(mtx_reco,2);
k1= k*nudgefactor*mtx_acq/mtx_reco;

kmag = max(sqrt(sum(abs(k1).^2,3)),[],2);
indices = kmag<0.5;
k1 = k1(indices,:,:);
nsamplesNew = size(k1,1);

nWraps = 15;
nangles = size(dd,3);
nAcqsInWrap = floor(nangles/nWraps);
nSpokesinIR=450;
n_IRs = floor(nangles/nSpokesinIR);


clear dd_1
clear k2
clear dcf1;
%dd_v1 = zeros(nvCh,nsamples,nSpokesinIR,210);
startSpoke = 1;
for nIR=1:floor(nangles/nSpokesinIR) 
    dd_1(:,:,:,nIR) = reshape(dd_v(:,indices,startSpoke:startSpoke+nSpokesinIR-1),[nvCh,nsamplesNew,nSpokesinIR,1]);
    k2(:,:,nIR,:) = k1(:,startSpoke:startSpoke+nSpokesinIR-1,:);
    dcf1(:,:,nIR) = dcf(indices,startSpoke:startSpoke+nSpokesinIR-1);

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
k2 = k2(:,:,2:end,:);
dcf1 = dcf1(:,:,2:end);


osf = 2; % oversampling: 1.5 1.25
wg = 3; % kernel width: 5 7
sw = 8; % parallel sectors' width: 12 16

sz = size(k2);
ktemp = reshape(k2,prod(sz(1:3)),3);
FT = gpuNUFFT(transpose(ktemp),ones(prod(sz(1:3)),1),osf,wg,sw,[1 1 1]*mtx_reco,[],true);
dtemp = reshape(dd_1,[nvCh,prod(sz(1:3))]);
dcftemp = dcf1; 
dcftemp = reshape(dcftemp,1,prod(sz(1:3)));
tic;coilImages = FT'*bsxfun(@times,dtemp,dcftemp)';toc
sos = sqrt(sum(abs(coilImages).^2,4));
imagesc(imrotate(squeeze(abs(sos(floor(mtx_reco/2),:,:))),-90),[0,2e6])
    
[cmap,mask] = giveCSM(coilImages,5,12,0.01);%
cmap = bsxfun(@times,cmap,mask);

SOS = sum(coilImages.*conj(cmap),4);
figure(1);imagesc((abs(SOS(:,:,floor(mtx_reco/2)))),[0,0.25*max(abs(SOS(:)))]);


% Computation of Atb for each bin

%indicesStart = [2,25,51,101,201];
indicesStart = [65,75,85,100,150];

numBins = length(indicesStart)
indicesEnd = indicesStart + [20,20,20,50,200];
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
    dcfbin = dcf1(:,indicesStart(i):indicesEnd(i),:); %dcftemp(:,1:3,:)=0;%dcftemp(:,[450:end],:)=0;
    dcfbin = reshape(dcfbin,1,nkBin);
    
    FT = gpuNUFFT(transpose(kbin),ones(size(dcfbin)),osf,wg,sw,[1 1 1]*mtx_reco,[],true);
    
    dd_bin = dd_1(:,:,indicesStart(i):indicesEnd(i),:);
    dd_bin = reshape(dd_bin,[nvCh,nkBin]); 
        
    Atb(:,:,:,:,i) = FT'*bsxfun(@times,dd_bin,dcfbin)'.*50./length(spokesInBin);
    sos(:,:,:,i) = sqrt(sum(abs(Atb(:,:,:,:,i)).^2,4));    
    kspaceWts(:,:,:,i) = giveNUFFTx2(dcfbin(:),FT,mtx_reco,kbin');
    kspaceWts(:,:,:,i) = (fftn(fftshift(kspaceWts(:,:,:,i))));

    imagesc(imrotate(squeeze(abs(sos(floor(mtx_reco/2),:,:,i))),-90)); drawnow
end
%

clear FT T G
T = FourierWtOpSeries(gpuArray(kspaceWts),gpuArray(cmap),true);
G = GradOp([mtx_reco,mtx_reco,mtx_reco,numBins],[2,2,2,1]);
M = @(x) T*x+1e-13*(G*x);

Atb1 = sum(bsxfun(@times,Atb,conj(cmap)),4);
tic;recon = pcg(M,gpuArray(Atb1(:)),1e-14,20,[],[],gpuArray(Atb1(:)));toc

recon = gather(recon);
recon = reshape(recon,mtx_reco,mtx_reco,mtx_reco,numBins);
disp('CG recovery done');
clear T;

sca =[0.4,0.4,0.4,0.4,0.4];
for i=1:size(recon,4)
    figure(i);tmp = squeeze(abs(recon(floor(mtx_reco/2),:,:,i)));
    imagesc(imrotate(squeeze(tmp),-90),[0,sca(i)*max(tmp(:))]); colormap(gray);drawnow
    axis('image'); axis off
end

sca =[0.1,0.2,0.2,0.2,0.2];
for i=1:size(recon,4)
    figure(i+10);tmp = squeeze(abs(recon(:,74,:,i)));
    imagesc(imrotate(squeeze(tmp),-90),[0,sca(i)*max(tmp(:))]); drawnow
    colormap(gray);axis('image'); axis off
end

%%
for i=1:5,
epsilon = 2e9;
G = GradOp3DWted([mtx_reco,mtx_reco,mtx_reco,numBins],[2,2,2,1]);
gradmag = G.gradMag(recon);
gradmag = sqrt(sum(gradmag,4));
gradmag = epsilon./(epsilon + gradmag);
G = GradOp3DWted([mtx_reco,mtx_reco,mtx_reco,numBins],[2,2,2,1],gradmag);
T = FourierWtOpSeries(kspaceWts,cmap,true);

M = @(x) T*x+5e-13*(G*x);
tic;recon = pcg(M,gpuArray(Atb1(:)),1e-14,10,[],[],gpuArray(recon(:)));toc
end