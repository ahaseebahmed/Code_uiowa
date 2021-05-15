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

d = '/Shared/lss_jcb/abdul/scan21Feb/P76800.7'
wf_name = '/Shared/lss_jcb/abdul/scan21Feb/radial3D_1H_fov200_mtx800_intlv147000_kdt6_gmax16_smax119_dur3p2_coca'

%d = '/Users/jcb/abdul_brain/scan17FebHuman/P87552_fullspoke.7'
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
nvCh = min(12,nCh);%min(find(s>0.5));
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

dcf = reshape(dcf,nsamples,nangles);
dcf(:,period:period:end)=0;
dcf(:,period-1:period:end)=0;
dcf(:,period-2:period:end)=0;

nSpokesToUse = 300*350;%size(dd,2)/nsamples;
dd1 = reshape(dd(:,1:nsamples*nSpokesToUse),[nvCh,nsamples,nSpokesToUse]);
k1 = reshape(k(1:nsamples*nSpokesToUse,:),[nsamples,nSpokesToUse,3]);
dcf1 = reshape(dcf(1:nsamples*nSpokesToUse),[nsamples,nSpokesToUse]);
%%
nudgefactor = 1.1;
mtx_reco = 600;
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

clear Atb;
%coilImages = zeros([mtx_reco*[1,1,1],nCh],class(dd_2));
Atb = zeros([mtx_reco*[1,1,1],nvCh],class(dd_2));
parfor i=1:nvCh,
    i
   Atb(:,:,:,i) = gridding3dsingle(k2,dd_2(i,:),dcf2,[1 1 1]*mtx_reco);
end

%osf = 2; wg = 3; sw = 8; % parallel sectors' width: 12 16
%FT = gpuNUFFT(transpose(k2),ones(size(dcf2)),osf,wg,sw,[1 1 1]*mtx_reco,[],true);
%clear coilImages;
%tic; coilImages = FT'*bsxfun(@times,dd_2',dcf2); toc
griddingrecon = sqrt(sum(abs(Atb).^2,4));
%sos = sqrt(sum(abs(coilImages).^2,4));
imagesc(abs(griddingrecon(:,:,mtx_reco/2)),[0,80])
%%

psf = gridding3dsingle(k2,ones(size(dcf2))',dcf2,[1 1 1]*mtx_reco);
H = (fftn(fftshift(psf)));


% Setup parameters (for example)
opts.rho_r   = 2;
opts.beta    = [1 1 1];
opts.print   = true;
opts.alpha   = 0.7;
opts.method  = 'l2';
opts.tol = 1e-8;
opts.max_itr = 20;
Hnew = H./abs(H(1,1,1));
opts.print = false;
%%
mu    = 5e2;
opts.rho_r = 1000;

out = zeros([mtx_reco*[1,1,1],nvCh],class(dd_2));
parfor i=1:nvCh,
    i
     temp = deconvtvl2(Atb(:,:,:,i),(Hnew),mu,opts);
     out(:,:,:,i) = temp.f;
end
%out = deconvtv(g, H, mu, opts);
tvrecon = sqrt(sum(abs(out).^2,4));
figure(2);imagesc(abs(tvrecon(:,:,mtx_reco/2)),[0,120])
