clear all
cd '/Users/jcb/abdul_brain/code'
addpath('/Users/jcb/MRSI/matlab_mns');
addpath('/Users/jcb/MRSI/MNS-Aug19/read_MR_MR27_7T');
addpath(genpath('/Users/jcb/medimgviewer'));

addpath(genpath('./../../gpuNUFFT'));

d = '/Users/jcb/abdul_brain/scan17FebHuman/P88576.7'
wf_name = '/Users/jcb/abdul_brain/scan17FebHuman/radial3D_1H_fov200_mtx800_intlv206550_kdt6_gmax16_smax119_dur3p2_coca'

do_single = true; % single mode to save memory
[dd,k,dcf,mtx_acq,nangles,fov,phi,theta]  = readradial(d,[],wf_name,[],[],[],[],...
    [],[],do_single,true);

%% Coil compressing the data
nCh = size(dd,1);
Rs = gather(real(dd*dd'));

[v,s] = eig(Rs);
s = diag(s);[s,i] = sort(s,'descend');
v = v(:,i);
s=s./sum(s);s = cumsum(s);
nvCh = min(6,nCh);%min(find(s>0.5));
V = zeros(nCh,nvCh);
V = v(:,1:nvCh);

  
dd_v = V'*dd;

dd = dd_v;
clear dd_v;

%%
k = squeeze(k);
load(wf_name);
nsamples = size(ks,2);
nangles = length(phi);

indices = find((phi==0) & (theta==0));
period = indices(5)-indices(4);

dcf(1:period:end)=0;
dcf(2:period:end)=0;
dcf(3:period:end)=0;

nSpokesToUse = 130800;
dd1 = reshape(dd(:,1:nsamples*nSpokesToUse),[nvCh,nsamples,nSpokesToUse]);
k1 = reshape(k(1:nsamples*nSpokesToUse,:),[nsamples,nSpokesToUse,3]);
dcf1 = reshape(dcf(1:nsamples*nSpokesToUse),[nsamples,nSpokesToUse]);
%%
nudgefactor = 1.1;
mtx_reco = 300;
mtx_reco = mtx_reco+mod(mtx_reco,2);
k2= k1*nudgefactor*mtx_acq/mtx_reco;

kmag = max(sqrt(sum(abs(k2).^2,3)),[],2);
indices = kmag<0.5;
k2 = k2(indices,:,:);
nsamplesNew = size(k2,1);
nanglesNew = size(k2,2);

k2 = reshape(k2,nsamplesNew*nanglesNew,3);
dcf2 = reshape(dcf1(indices,:,:),nsamplesNew*nanglesNew,1);
dd_2 = reshape(dd1(:,indices,:),nvCh,nsamplesNew*nanglesNew);
%dcf1 = reshape(dcf1(indices,:,:),nsamplesNew*nanglesNew,

clear sos;
%coilImages = zeros([mtx_reco*[1,1,1],nCh],class(dd_2));
sos = zeros(mtx_reco*[1,1,1],class(dd_2));
for i=1:1,
    i
   temp = gridding3dsingle(k2,dd_2(i,:),dcf2,[1 1 1]*mtx_reco);
   sos = sos + abs(temp).^2;
end

%osf = 2; wg = 3; sw = 8; % parallel sectors' width: 12 16
%FT = gpuNUFFT(transpose(k2),ones(size(dcf2)),osf,wg,sw,[1 1 1]*mtx_reco,[],true);
%clear coilImages;
%tic; coilImages = FT'*bsxfun(@times,dd_2',dcf2); toc
sos = sqrt(sos);
%sos = sqrt(sum(abs(coilImages).^2,4));
imagesc(abs(sos(:,:,200)),[0,100])

%%
load(wf_name);
nsamples = size(ks,2);
nangles = length(phi);

nCh = size(dd,1);
dd = reshape(dd,nCh,nsamples,nangles);
k = reshape(k,nsamples,nangles,3);
dcf = reshape(dcf,nsamples,nangles);

indices = find((phi==0) & (theta==0));
period = indices(5)-indices(4);

ind = ones(size(dd,2),size(dd,3));

%% Sorting data
nvCh = nCh;
nWraps = 15;
nAcqsInWrap = floor(nangles/nWraps);
nSpokesinIR=375;
n_IRs = floor(nangles/nSpokesinIR);

dd_1 = zeros(nvCh,nsamples,nSpokesinIR,floor(nangles/nSpokesinIR),class(dd));
dcf1 = zeros(nsamples,nSpokesinIR,floor(nangles/nSpokesinIR),class(dcf));
k1 = complex(zeros(nsamples,nSpokesinIR,floor(nangles/nSpokesinIR),3));


startSpoke = 1;
for nIR=1:400,%floor(nangles/nSpokesinIR)-2 % more noise in later images; motion ??
    nIR
    pause(0.1)
    dd_1(:,:,:,nIR) = reshape(dd(:,:,startSpoke:startSpoke+nSpokesinIR-1),[nvCh,nsamples,nSpokesinIR,1]);
    k1(:,:,nIR,:) = k(:,startSpoke:startSpoke+nSpokesinIR-1,:);
    dcf1(:,:,nIR) = dcf(:,startSpoke:startSpoke+nSpokesinIR-1);

    startSpoke = startSpoke+nSpokesinIR;
    
    if(mod(startSpoke+nSpokesinIR,nAcqsInWrap)<=nSpokesinIR)
        startSpoke = startSpoke+2*nSpokesinIR-mod(startSpoke+nSpokesinIR,nAcqsInWrap);
    end
    if(startSpoke>nangles-50*nSpokesinIR)
        break
    end
end

dd_1 = dd_1(:,:,:,2:nIR);
k1 = k1(:,:,2:nIR,:);
dcf1 = dcf1(:,:,2:nIR);

%%
nudgefactor = 1.2;
mtx_reco = 300;
mtx_reco = mtx_reco+mod(mtx_reco,2);
k2= k1*nudgefactor*mtx_acq/mtx_reco;

kmag = max(max(sqrt(sum(abs(k2).^2,4)),[],2),[],3);
indices = kmag<0.5;
k2 = k2(indices,:,:,:);
nsamplesNew = size(k2,1);
nanglesNew = size(k2,2)*size(k2,3);

k2 = reshape(k2,nsamplesNew*nanglesNew,3);
dcf2 = reshape(dcf1(indices,:,:),nsamplesNew*nanglesNew,1);
dd_2 = reshape(dd_1(:,indices,:,:),nCh,nsamplesNew*nanglesNew);
%dcf1 = reshape(dcf1(indices,:,:),nsamplesNew*nanglesNew,

clear sos;
%coilImages = zeros([mtx_reco*[1,1,1],nCh],class(dd_2));
sos = zeros(mtx_reco*[1,1,1],class(dd_2));
for i=1:nCh,
   temp = gridding3dsingle(k2,dd_2(i,:),dcf2,[1 1 1]*mtx_reco);
   sos = sos + abs(temp).^2;
end

%osf = 2; wg = 3; sw = 8; % parallel sectors' width: 12 16
%FT = gpuNUFFT(transpose(k2),ones(size(dcf2)),osf,wg,sw,[1 1 1]*mtx_reco,[],true);
%clear coilImages;
%tic; coilImages = FT'*bsxfun(@times,dd_2',dcf2); toc
sos = sqrt(sos);
%sos = sqrt(sum(abs(coilImages).^2,4));
imagesc(abs(sos(:,:,200)),[0,100])
%% Initial reconstruction at lower resolution
mtx_reco = 100;
load(wf_name);

clear params;

indices = find((phi==0) & (theta==0));
period = indices(5)-indices(4);

dcf(:,1:period:end)=0;
dcf(:,2:period:end)=0;
dcf(:,3:period:end)=0;

params.dcf = dcf;
params.dd = reshape(dd_1,nCh,nsamples,nIR*nSpokesinIR);
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

%mask = abs(sos)<0.22*max(sos(:));
%coilImages = bsxfun(@times,coilImages,mask);
%[denoised_norm,biasfield] = giveNormalizedImage(abs(test1),.15,1);

disp('Coil Compression');
tmp = coilImages;
%tmp = coilImages(:,:,50:end,:);
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


%% Final recon at 340 matrix size
mtx_reco = 600

load(wf_name);

clear params;
params.dcf = dcf(:,1:375*400);
params.dd = dd(:,:,1:375*400);
params.phi = phi(1:375*400);
params.theta = theta(1:375*400);
params.nS = size(ks,2);
params.ks = ks*1.2*mtx_acq/mtx_reco; 
params.indices = squeeze(find(sum(abs(params.ks).^2,3)<0.25));

params.ks = params.ks(:,params.indices,:);
params.mtx_reco = mtx_reco;

tform = affine3d(eye(4));
params = transformKspaceAndData(tform,params);  
sos = zeros(mtx_reco*[1,1,1]);
for i=1:nCh,
    i
    temp = params.FT'*bsxfun(@times,params.dd(i,:)',(params.dcf(:)));
    sos = sos + abs(temp).^2;
end
sos = sqrt(sos);
%[cmap,mask] = giveCSM(coilImages,5,14,0.07);
%cmap = bsxfun(@times,cmap,mask);
%Atb = sum(coilImages.*conj(cmap),4);

%%
indnew = ind(params.indices,:); 
indnew(:,1:period:end)=0;
indnew(:,2:period:end)=0;
indnew(:,3:period:end)=0;
indnew = indnew(:).*params.dcf(:);

kspaceWts = giveNUFFTx2(indnew(:),params.FT,params.mtx_reco,params.k');
kspaceWts = (fftn(fftshift(kspaceWts)));

clear A
A = FourierWtOpSeries(kspaceWts,ones(mtx_reco*[1,1,1]),true);
G = TVOp(size(Atb),[1,1,1],true);

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
