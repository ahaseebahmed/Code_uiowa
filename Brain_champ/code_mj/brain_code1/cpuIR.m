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


d='/Users/jcb/abdul_brain/31Jan20/highRes/P53248_1ResHuman.7';
%d='./../../ScannerCodes/reconstruction/scan31Jan/P69632_1ResHuman2.7';

wf_name = '/Users/jcb/abdul_brain/31Jan20/highRes/radial3D_1H_fov200_mtx200_intlv126300_kdt4_gmax23_smax120_dur0p8_coca.mat';
%wf_name = 'radial3D_1H_fov200_mtx400_intlv63200_kdt8_gmax33_smax118_dur5p2_fsca.mat';

[dd,k,dcf,mtx_acq,nangles,fov,phi,theta]  = readradial(d,[],wf_name);

nsamples = length(dcf)/nangles;
c_ind=[2:7,10:15,18,19,22,23,26:27,30,31];
dd=dd(c_ind,:);
nCh = size(dd,1);
dd = reshape(dd,nCh,nsamples,nangles);
k = reshape(k,nsamples,nangles,3);
dcf = reshape(dcf,nsamples,nangles);

ind = ones(size(dd,2),size(dd,3));
% for i=1:1140:nangles,
%     dd(:,:,i:i+2)=0;
%     ind(:,i:i+2)=0;
% end
% 
% for i=1:600:nangles,
%     dd(:,:,i:i+2)=0;
%     ind(:,i:i+2)=0;
% end
%%
disp('Coil Compression');
tmp = (dd(:,1:end,:));
tmp = reshape(tmp,nCh,size(tmp,2)*nangles);
Rs = (real(tmp*tmp'));

  [v,s] = eig(Rs);
  s = diag(s);[s,i] = sort(s,'descend');
  v = v(:,i);
  s=s./sum(s);s = cumsum(s);
  nvCh = min(8,nCh);%min(find(s>0.9));

  dd_v = (v(:,1:nvCh)'*reshape((dd),nCh,nsamples*nangles));  
  dd_v = reshape(dd_v,nvCh,nsamples,nangles);
%% Scanner based sorting
mtx_reco=240;
nangles1 = size(dd_v,3);
n_SpokesinIR=600;
n_IRs = floor(nangles1/n_SpokesinIR);

dd_v1 = reshape(dd_v(:,:,1:n_IRs*n_SpokesinIR),nvCh,nsamples,n_SpokesinIR,n_IRs);
k1 = reshape(k(:,1:n_IRs*n_SpokesinIR,:),nsamples,n_SpokesinIR,n_IRs,3);
dcf1 = reshape(dcf(:,1:n_IRs*n_SpokesinIR),nsamples,n_SpokesinIR,n_IRs);

%%
mtx_reco = 240;
nSamplesPerBin = 75;
nSamplesIncrement = 25;

indicesStart = [4:15:300];
indicesEnd = indicesStart+nSamplesPerBin;
indicesEnd(end)=n_SpokesinIR;

osf = 2; % oversampling: 1.5 1.25
wg = 3; % kernel width: 5 7
sw = 8; % parallel sectors' width: 12 16
sz = size(k1) ;nk = prod(sz(1:3));
FT = gpuNUFFT(reshape(k1,nk,sz(4))',ones(nk,1),osf,wg,sw,[1 1 1]*mtx_reco,[],true);
dd_v1 = transpose(reshape(dd_v1,nvCh,nsamples*n_SpokesinIR*n_IRs));

p = {};
for i=1:8,%length(indicesStart),
    spokesInBin = indicesStart(i):indicesEnd(i);
    p{i}.invTime = mean(spokesInBin)*5;
    p{i}.dcf = zeros(size(dcf1));
    p{i}.dcf(:,indicesStart(i):indicesEnd(i),:) =  dcf1(:,indicesStart(i):indicesEnd(i),:);
    p{i}.dcf = p{i}.dcf(:);
    tic;p{i}.recon = FT'*bsxfun(@times,dd_v1,p{i}.dcf);toc
end
%%
    ktemp = reshape(k1,nsamples*n_SpokesinIR*size(k1,3),3);
    FT = gpuNUFFT(transpose(ktemp),ones(nsamples*n_SpokesinIR*size(k1,3),1),osf,wg,sw,[1 1 1]*mtx_reco,[],true);
    dtemp = reshape(dd_v1,nvCh,nsamples*n_SpokesinIR*size(k1,3));
    dcftemp = dcf1; dcftemp(:,1:3,:)=0;dcftemp(:,[1:150,450:end],:)=0;
    dcftemp = reshape(dcftemp,1,nsamples*n_SpokesinIR*size(k1,3));
    coilImages = FT'*bsxfun(@times,dtemp,dcftemp)';
    SOS = sqrt(sum(abs(coilImages).^2,4));
    imagesc(flipud(abs(SOS(:,:,100))),[0,5e6]);

    %%
for i=1:length(p)
    nrows = 2;
    ncols = ceil(length(p)/nrows);
   
    q = sqrt(sum(abs(p{i}.recon).^2,4));
    q = q./length(p{i}.dcf);
    figure(1);subplot(nrows,ncols,i), imagesc(flipud(abs(q(:,:,120))),[0,.05]); 
    colormap(gray);
    title(num2str(p{i}.invTime))
    axis off; axis('image');drawnow;
end

%% Sorting for IR Reconstrcution
nn=reshape(dd_v,[nvCh,nsamples,frm,npt,nrepp]);
kk=reshape(k,[nsamples,frm,npt,nrepp,3]);
dcf=dcf(:,1:nrepp*n_IR);
dcf1=reshape(dcf,[nsamples,frm,npt,nrepp]);
tmp=nn(:,:,:,1:4,:);
kkt1=kk(:,:,1:4,:,:);
dcft1=dcf1(:,:,1:4,:);


tmp=reshape(tmp,[nvCh,nsamples,frm*npt/2,nrepp]);
kkt1=reshape(kkt1,[nsamples,frm*npt/2,nrepp,3]);
dcft1=reshape(dcft1,[nsamples,frm*npt/2,nrepp]);

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

tmp1=nn(:,:,:,5:8,:);
kkt2=kk(:,:,5:8,:,:);
dcft2=dcf1(:,:,5:8,:);
tmp1=reshape(tmp1,[nvCh,nsamples,frm*npt/2,nrepp]);
kkt2=reshape(kkt2,[nsamples,frm*npt/2,nrepp,3]);
dcft2=reshape(dcft2,[nsamples,frm*npt/2,nrepp]);


% nn=cat(4,tmp2,tmp1);
% kk=cat(3,kkt,kkt2);
% dcf1=cat(3,dcft,dcft2);
%tmp3(:,:,:,:)=
      
%% RECONSTRUCTION
mtx_reco = 200;
kkt = kkt*1.1*mtx_acq/mtx_reco; 

if(mtx_reco>mtx_acq)
    nS = nsamples;
else
    nS = floor(nsamples*mtx_reco/mtx_acq);
end

for ij=1:npt1+1

    if ij<=npt1
        nn1=squeeze(tmp2(:,1:nS,:,ij,:));
        kk1=squeeze(kkt(1:nS,:,ij,:,:));
        dcf=squeeze(dcft(1:nS,:,ij,:));
        k1_new=reshape(kk1,[nsamples*wind*nrepp,3]);
        dcf_new=reshape(dcf,[nsamples*wind*nrepp,1]);
        dd_new=reshape(nn1,[nvCh,nsamples*wind*nrepp]);
    else
        k1_new=reshape(kkt2,[nsamples*frm*(npt/2)*nrepp,3]);
        dcf_new=reshape(dcft2,[nsamples*frm*nrepp*(npt/2),1]);
        dd_new=reshape(tmp1,[nvCh,nsamples*frm*(npt/2)*nrepp]);
    end


init_recon = zeros(mtx_reco,mtx_reco,mtx_reco,nvCh);
tic
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
ind_new=ones(size(k1_new,1),1);
kspaceWts = gridding3dsingle(k1_new,ind_new(:)',((dcf_new)),[1 1 1]*mtx_reco);
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
niter = 300;pdenoised = 0*Atb;lambda =0.2;%p1 = pdenoised;
%niter = 5;pdenoised = 0*Atb;lambda =0.01;%p1 = pdenoised;

T = FourierWtOp(kspaceWts,cmapnew,true);
G = TVOp(size(kspaceWts),[1,1,1],true);

M = @(x) T*x + lambda*(G*x);
 
tic;p1 = pcg(M,Atb(:)+lambda*gpuArray(pdenoised(:)),1e-14,niter,[],[],gpuArray(pdenoised(:)));toc
p1 = gather(p1);
p(:,:,:,ij) = reshape(p1,mtx_reco,mtx_reco,mtx_reco);
if ij>npt1
    p(:,:,:,ij)=sqrt(sum(init_recon.*conj(init_recon),4));
end

disp('CG recovery done');
clear T;
end


% figure(1);imagesc(abs(p(:,:,100,1)));drawnow; colormap gray;
% figure(1);colormap gray;
% for i=1:400
%     imagesc(abs(pp(:,:,i)));pause(0.1);
% end
% 
%     imagesc_ind3d(p,'',[min(abs(p(:))),15],'',false,true);
% 
% %%
% figure;
% dd1=squeeze(p(:,:,100,:));
% dd1=reshape(abs(dd1),[200,200*10]);
% imagesc((dd1(:,:)));
% colormap hot
% 
% 
figure;
for j=100
    j
for i=1:20;colormap gray;
    subplot(4,5,i);imagesc((squeeze(abs(p(:,:,j,i+20)))));
    %subplot(1,4,i);imshow((squeeze(abs(dd(50,:,:,i)))));
end
pause();

end