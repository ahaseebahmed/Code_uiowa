clear all
% cd '/Users/jcb/abdul_brain/code'
% addpath('/Users/jcb/MRSI/matlab_mns');
% addpath('/Users/jcb/MRSI/MNS-Aug19/read_MR_MR27_7T');
% addpath(genpath('/Users/jcb/medimgviewer'));
% 
 addpath(genpath('./../gpuNUFFT'));
 
%addpath(genpath('./../../../Medical Image Reader and Viewer'));
% addpath(genpath('/Users/jcb/nfft2/matlab/nfft'));
% addpath('/Users/jcb/bm4d/');
% addpath(genpath('/Users/jcb/bm4d/MRIDenoisingPackage_r01_pcode'));
addpath(genpath('./../../../ScannerCodes/read_MR_MR27_7T'));
addpath(genpath('./../../../ScannerCodes/matlab'));
%d='./../../../ScannerCodes/vds_7T/P87552_fullspoke.7';
%wf_name = './../../../ScannerCodes/reconstruction/scan17Feb/radial3D_1H_fov200_mtx800_intlv130800_kdt6_gmax33_smax119_dur8_fsca.mat';
% 
% 
% %[dd,k,dcf,mtx_acq,nangles,fov,phi,theta]  = readradialradial3D_1H_fov200_mtx800_intlv130800_kdt6_gmax33_smax119_dur8_fsca(d,[],wf_name);
%[dd,k,dcf,mtx_acq,nangles,fov,phi,theta]  = readradial1(d,[],wf_name);

%[dd,k,dcf,mtx_acq,nangles,fov,phi,theta]  = readradial(d,[],wf_name);
load('./../res0.2qg.mat');
nsamples = length(dcf)/nangles;
%c_ind=[2:7,10:15,18,19,22,23,26:27,30,31];
%dd=dd(c_ind,:);
nCh = size(dd,1);
dd = reshape(dd,nCh,nsamples,nangles);
k = reshape(k,nsamples,nangles,3);
dcf = reshape(dcf,nsamples,nangles);
ind = ones(size(dd,2),size(dd,3));

for i=1:450:nangles,
    dd(:,:,i:i+2)=0;
    ind(:,i:i+2)=0;
end

%% Initial reconstruction

% nudgefactor = 1.1;
% mtx_reco = mtx_acq;
% mtx_reco = mtx_reco+mod(mtx_reco,2);
% mtx_reco=400;
% 
% osf = 2; wg = 3; sw = 8; 
% k1= k*nudgefactor*mtx_acq/mtx_reco;
% nangles = size(dd,3);
% 
% kmag = max(sqrt(sum(abs(k1).^2,3)),[],2);
% indices = kmag<0.5;
% k1 = k1(indices,:,:);
% nsamplesNew = size(k1,1);
% 
% sz = size(k1);
% ktemp = reshape(k1,prod(sz(1:2)),3);
% 
% FT = gpuNUFFT(transpose(ktemp),ones(prod(sz(1:2)),1),osf,wg,sw,[1 1 1]*mtx_reco,[],true);
% dtemp = reshape(dd(:,indices,:),[nCh,nsamplesNew*nangles]);
% dcftemp = reshape(dcf(indices,:),[1,nsamplesNew*nangles]);
% tic;coilImages = FT'*bsxfun(@times,dtemp,dcftemp)';toc
% sos = sqrt(sum(abs(coilImages).^2,4));
% imagesc(imrotate(squeeze(abs(sos(floor(mtx_reco/2),:,:))),-90))

%%
disp('Coil Compression');
tmp = (dd(:,:,:));
tmp = reshape(dd,nCh,nsamples*nangles);
Rs = (real(tmp*tmp'));

  [v,s] = eig(Rs);
  s = diag(s);[s,i] = sort(s,'descend');
  v = v(:,i);
  s=s./sum(s);s = cumsum(s);
  nvCh = min(4,nCh);%min(find(s>0.7));

  dd_v = (v(:,1:nvCh)'*reshape((dd),nCh,nsamples*nangles)); 
  dd_v = reshape(dd_v,nvCh,nsamples,nangles);
  
%% Scanner based sorting
% nudgefactor = 1.2;
% mtx_reco = 258;
% mtx_reco = mtx_reco+mod(mtx_reco,2);
% k1= k*nudgefactor*mtx_acq/mtx_reco;
% 
% kmag = max(sqrt(sum(abs(k1).^2,3)),[],2);
% indices = kmag<0.5;
% k1 = k1(indices,:,:);
% nsamplesNew = size(k1,1);
% 
% nWraps = 15;
% nangles = size(dd,3);
% nAcqsInWrap = floor(nangles/nWraps);
% nSpokesinIR=450;
% n_IRs = floor(nangles/nSpokesinIR);


clear dd_1
clear k2
clear dcf1;
%dd_v1 = zeros(nvCh,nsamples,nSpokesinIR,210);
startSpoke = 1;
% for nIR=1:floor(nangles/nSpokesinIR) 
%     dd_1(:,:,:,nIR) = reshape(dd_v(:,indices,startSpoke:startSpoke+nSpokesinIR-1),[nvCh,nsamplesNew,nSpokesinIR,1]);
%     k2(:,:,nIR,:) = k1(:,startSpoke:startSpoke+nSpokesinIR-1,:);
%     dcf1(:,:,nIR) = dcf(indices,startSpoke:startSpoke+nSpokesinIR-1);
% 
%     startSpoke = startSpoke+nSpokesinIR;
%     
%     if(mod(startSpoke+nSpokesinIR,nAcqsInWrap)<=nSpokesinIR)
%         startSpoke = startSpoke+2*nSpokesinIR-mod(startSpoke+nSpokesinIR,nAcqsInWrap);
%     end
%     if(startSpoke>nangles)
%         break
%     end
% end

n_IR=375;
n4=15;
n1=nangles/n4;
%nrepp=floor(nangles/n_IR);
dd_v=reshape(dd_v,[nvCh,nsamples,n1,n4]);
k=reshape(k,[nsamples,n1,n4,3]);
dcf=reshape(dcf,[nsamples,n1,n4]);
t11=floor(n1/n_IR);
dd_v=dd_v(:,:,n_IR+1:n_IR*t11,:);

k=k(:,n_IR+1:n_IR*t11,:,:);
dcf=dcf(:,n_IR+1:n_IR*t11,:);
t12=size(dd_v,3);
dd_v=dd_v(:,:,1:t12*n4);
k=reshape(k,[nsamples,t12*n4,3]);
dcf=dcf(:,:);
frm=75;
npt=5;
nrepp=floor(t12*n4/n_IR);
dd_v=dd_v(:,:,1:nrepp*n_IR);
k=k(:,1:nrepp*n_IR,:);
dcf=dcf(:,1:nrepp*n_IR);

%% Sorting for IR Reconstrcution
first_rec_pts=4;

nn=reshape(dd_v,[nvCh,nsamples,frm,npt,nrepp]);
kk=reshape(k,[nsamples,frm,npt,nrepp,3]);
dcf=dcf(:,1:nrepp*n_IR);
dcf1=reshape(dcf,[nsamples,frm,npt,nrepp]);
tmp=nn(:,:,:,1:first_rec_pts,:);
kkt1=kk(:,:,1:first_rec_pts,:,:);
dcft1=dcf1(:,:,1:first_rec_pts,:);

tmp=reshape(tmp,[nvCh,nsamples,frm*first_rec_pts,nrepp]);
kkt1=reshape(kkt1,[nsamples,frm*first_rec_pts,nrepp,3]);
dcft1=reshape(dcft1,[nsamples,frm*first_rec_pts,nrepp]);

wind=75;
step=25;
npt1=((size(tmp,3)-wind)/step)+1;
tmp2=zeros(nvCh,nsamples,wind,npt1,nrepp);
kkt=zeros(nsamples,wind,npt1,nrepp,3);
dcft=zeros(nsamples,wind,npt1,nrepp);
for i=1:npt1
    tmp2(:,:,:,i,:)=tmp(:,:,((i-1)*step)+1:((i-1)*step)+wind,:);
    kkt(:,:,i,:,:)=kkt1(:,((i-1)*step)+1:((i-1)*step)+wind,:,:);
    dcft(:,:,i,:)=dcft1(:,((i-1)*step)+1:((i-1)*step)+wind,:);
end

% total_sec_rec=1;
% tmp1=nn(:,:,:,first_rec_pts+1:end,:);
% kkt2=kk(:,:,first_rec_pts+1:end,:,:);
% dcft2=dcf1(:,:,first_rec_pts+1:end,:);
% tmp1=reshape(tmp1,[nvCh,nsamples,frm*total_sec_rec,nrepp]);
% kkt2=reshape(kkt2,[nsamples,frm*total_sec_rec,nrepp,3]);
% dcft2=reshape(dcft2,[nsamples,frm*total_sec_rec,nrepp]);
%%

mtx_reco = 100;
kkt = kkt*1.1*mtx_acq/mtx_reco; 

if(mtx_reco>mtx_acq)
    nS = nsamples;
else
    nS = floor(nsamples*mtx_reco/mtx_acq);
end

str=1;ij=1;
nn1=squeeze(tmp2(:,str:nS+str-1,:,:,:));
kk1=squeeze(kkt(str:nS+str-1,:,:,:,:));
dcf=squeeze(dcft(1:nS,:,:,:));
k1_new=reshape(kk1,[nS*wind*nrepp,npt1,3]);
dcf_new=reshape(dcf,[nS*wind*nrepp,npt1,1]);
dd_new=reshape(nn1,[nvCh,nS*wind*nrepp,npt1]);


%%
osf = 2; % oversampling: 1.5 1.25
wg = 3; % kernel width: 5 7
sw = 8; % parallel sectors' width: 12 16

sz = size(k1_new);
ktemp = reshape(k1_new,prod(sz(1:2)),3);
% FT = gpuNUFFT((ktemp)',ones(prod(sz(1:2)),1),osf,wg,sw,[1 1 1]*mtx_reco,[],true);
dtemp = reshape(dd_new,[nvCh,prod(sz(1:2))]);
% dcftemp = dcf_new(:); 
dcftemp = reshape(dcf_new,1,prod(sz(1:2)));
% tic;coilImages = FT'*bsxfun(@times,dtemp',dcftemp);toc
%sos = sqrt(sum(abs(coilImages).^2,4));
%imagesc(imrotate(squeeze(abs(sos(floor(mtx_reco/2),:,:))),-90),[0,2e6])
% init_recon = zeros(mtx_reco,mtx_reco,mtx_reco,nvCh);
tic
parfor i=1:nvCh,
init_recon(:,:,:,i) = gridding3dsingle(ktemp,dtemp(i,:),dcftemp,[1 1 1]*mtx_reco);
end
toc

% [cmap,mask] = giveCSM(init_recon,4,12);
% cmap = bsxfun(@times,cmap,mask);
% 
% % Heurestic; scaling the cmaps by sos of cmaps to normalize
% sos = sqrt(sum(abs(cmap).^2,4));
% sos = sos+not(mask);
% cmapnew = bsxfun(@times,cmap,1./sos);

[cmap,mask] = giveCSM(init_recon,5,12,0.01);%
cmap = bsxfun(@times,cmap,mask);

% SOS = sum(coilImages.*conj(cmap),4);
% figure(1);imagesc((abs(SOS(:,:,floor(mtx_reco/2)))),[0,0.25*max(abs(SOS(:)))]);


%% Computation of Atb for each bin

numBins=4;
bins=[1:4];
Atb = zeros(mtx_reco,mtx_reco,mtx_reco,nvCh,numBins);
kspaceWts = zeros(2*mtx_reco,2*mtx_reco,2*mtx_reco,numBins);

for i=1:numBins,
    i
    tmp=squeeze(k1_new(:,bins(i),:))';
    %FT = gpuNUFFT(tmp,ones(size(tmp,2),1),osf,wg,sw,[1 1 1]*mtx_reco,[],true);
    tmp1=bsxfun(@times,dd_new(:,:,bins(i)),dcf_new(:,bins(i))');
    %Atb(:,:,:,:,i) = FT'*bsxfun(@times,tmp1',dcf_new(:,i));
    Atb(:,:,:,:,i) = gridding3dsingle(tmp,tmp1,ones(size(dcf_new(:,bins(i)))),[1 1 1]*mtx_reco);
    %sos(:,:,:,i) = sqrt(sum(abs(Atb(:,:,:,:,i)).^2,4));    
    kspaceWts(:,:,:,i) = giveNUFFTx2_updated(dcf_new(:,bins(i)),mtx_reco,squeeze(k1_new(:,bins(i),:)));
    %kspaceWts(:,:,:,i) = giveNUFFTx2(dcf_new(:,i),FT,mtx_reco,squeeze(k1_new(:,i,:))');
    kspaceWts(:,:,:,i) = (fftn(fftshift(kspaceWts(:,:,:,i))));
end


clear FT
T = FourierWtOpSeries(kspaceWts,cmap,true);
G = GradOp([mtx_reco,mtx_reco,mtx_reco,numBins],[2,2,2,1]);

M = @(x) T*x+5e-13*(G*x);

Atb1 = sum(bsxfun(@times,Atb,conj(cmap)),4);
tic;recon = pcg(M,gpuArray(Atb1(:)),1e-14,5,[],[],gpuArray(Atb1(:)));toc

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

sca =[0.1,0.1,0.1,0.1,0.1];
for i=1:size(recon,4)
    figure(i+10);tmp = squeeze(abs(recon(50:end-50,154,50:end-50,i)));
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