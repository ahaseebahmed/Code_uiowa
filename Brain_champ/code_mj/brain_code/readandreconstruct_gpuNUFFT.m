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

%d = '/Users/jcb/abdul_brain/scan17FebHuman/P88576.7'
%wf_name = '/Users/jcb/abdul_brain/scan17FebHuman/radial3D_1H_fov200_mtx800_intlv206550_kdt6_gmax16_smax119_dur3p2_coca'

d = '/Shared/lss_jcb/abdul/scan21Feb/P76800.7'
wf_name = '/Shared/lss_jcb/abdul/scan21Feb/radial3D_1H_fov200_mtx800_intlv147000_kdt6_gmax16_smax119_dur3p2_coca'

%d = '/Shared/lss_jcb/abdul/scan21Feb/P76800.7'
%wf_name = '/Shared/lss_jcb/abdul/scan21Feb/radial3D_1H_fov200_mtx800_intlv147000_kdt6_gmax16_smax119_dur3p2_coca'

%wf_name = '/Users/jcb/abdul_brain/scan17FebHuman/radial3D_1H_fov200_mtx800_intlv130800_kdt6_gmax33_smax119_dur8_fsca_fullspoke'

%d='/Users/jcb/abdul_brain/31Jan20/P97792_0.75Res_IR.7';
%wf_name = '/Users/jcb/abdul_brain/31Jan20/radial3D_1H_fov192_mtx256_intlv206625_kdt4_gmax24_smax120_dur1_coca.mat';

%d='/Users/jcb/abdul_brain/31Jan20/highRes/P53248_1ResHuman.7';
%wf_name = '/Users/jcb/abdul_brain/31Jan20/highRes/radial3D_1H_fov200_mtx200_intlv126300_kdt4_gmax23_smax120_dur0p8_coca.mat';

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
nvCh = min(4,nCh);%min(find(s>0.5));
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
period = indices(5)-indices(4);5

dcf = reshape(dcf,nsamples,nangles);
dcf(:,period:period:end)=0;
dcf(:,period-1:period:end)=0;
dcf(:,period-2:period:end)=0;


nSpokesToUse = size(dd,2)/nsamples;
dd1 = reshape(dd(:,1:nsamples*nSpokesToUse),[nvCh,nsamples,nSpokesToUse]);
k1 = reshape(k(1:nsamples*nSpokesToUse,:),[nsamples,nSpokesToUse,3]);
dcf1 = reshape(dcf(1:nsamples*nSpokesToUse),[nsamples,nSpokesToUse]);
dcf1(:,51:50:end)=0;
dcf1(:,52:50:end)=0;
%dcf1(:,53:50:end)=0;

%%
mtx_reco = 500;
tosf = 1.25; % oversampling: 1.5 1.25
wg = 3; % kernel width: 5 7
sw = 8; % parallel sectors' width: 12 16

nudgefactor = 1.1;
mtx_os = floor(mtx_reco*tosf);
mtx_os = mtx_os +8 - mod(mtx_os,8);
osf = (mtx_os/mtx_reco);

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


dcf2(isnan(dcf2))=0;
FT = gpuNUFFT(transpose(k2),1+0*col(dcf2),osf,wg,sw,[1 1 1]*mtx_reco,[],true);

Atb = FT'*bsxfun(@times,dd_2,dcf2')';
[cmap,mask] = giveCSM(Atb,4,12);
cmap = bsxfun(@times,cmap,mask);

FT = gpuNUFFT(transpose(k2),1+0*col(dcf2),osf,wg,sw,[1 1 1]*mtx_reco,cmap,true);
sos = FT'*bsxfun(@times,dd_2,dcf2')';
figure(1);imagesc(abs(sos(:,:,floor(mtx_reco/2))))

%%
T = TVOp(mtx_reco*[1,1,1],[1,1,1],false);
A = @(x) giveATA(x,FT,dcf2,mtx_reco*[1,1,1]);
Q = @(x) A(x) + 1e-16*(T*x);

tic;test = pcg(Q,sos(:),1e-4,10);toc
test = reshape(test,mtx_reco*[1,1,1]);

imagesc(abs(test(:,:,150)))
%%
