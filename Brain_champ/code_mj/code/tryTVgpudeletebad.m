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

d = '/Shared/lss_jcb/abdul/scan27Feb/P82432.7'
wf_name = '/Shared/lss_jcb/abdul/scan27Feb/radial3D_1H_fov200_mtx600_intlv147000_kdt8_gmax12_smax117_dur3p2_coca'

%d = '/Shared/lss_jcb/abdul/scan21Feb/P76800.7'
%wf_name = '/Shared/lss_jcb/abdul/scan21Feb/radial3D_1H_fov200_mtx800_intlv147000_kdt6_gmax16_smax119_dur3p2_coca'

%wf_name = '/Users/jcb/abdul_brain/scan17FebHuman/radial3D_1H_fov200_mtx800_intlv130800_kdt6_gmax33_smax119_dur8_fsca_fullspoke'

%d='/Users/jcb/abdul_brain/31Jan20/P97792_0.75Res_IR.7';
%wf_name = '/Users/jcb/abdul_brain/31Jan20/radial3D_1H_fov192_mtx256_intlv206625_kdt4_gmax24_smax120_dur1_coca.mat';

%d='/Users/jcb/abdul_brain/31Jan20/highRes/P53248_1ResHuman.7';
%wf_name = '/Users/jcb/abdul_brain/31Jan20/highRes/radial3D_1H_fov200_mtx200_intlv126300_kdt4_gmax23_smax120_dur0p8_coca.mat';

do_single = true; % single mode to save memory
[dd,k,dcf,mtx_acq,nangles,fov,phi,theta]  = readradial(d,[],wf_name,[],[5],[],[],...
    [],[],do_single,true);

%% Coil compressing the data
nCh = size(dd,1);
Rs = gather(real(dd*dd'));

[v,s] = eig(Rs);
s = diag(s);[s,i] = sort(s,'descend');
v = v(:,i);
s=s./sum(s);s = cumsum(s);
nvCh = min(8,nCh);%min(find(s>0.5));
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

test = squeeze(sum(abs(dd(1,100:end,:))));
[mx,worst] = max(test);
badindices = find(test>0.05*mx);

nSpokesToUse = size(dd,2)/nsamples;
dd1 = reshape(dd(:,1:nsamples*nSpokesToUse),[nvCh,nsamples,nSpokesToUse]);
k1 = reshape(k(1:nsamples*nSpokesToUse,:),[nsamples,nSpokesToUse,3]);
dcf1 = reshape(dcf(1:nsamples*nSpokesToUse),[nsamples,nSpokesToUse]);
dcf1(:,badindices)=0;

%dcf1 = reshape(dcf1,[nsamples,350,420]);
%dcf1(:,1:40,:)=0;
%dcf1(:,100:end,:)=0;

%%
nudgefactor = 1.1;
mtx_reco = 400;

tosf = 1.25; % oversampling: 1.5 1.25
wg = 3; % kernel width: 5 7
sw = 8; % parallel sectors' width: 12 16
mtx_os = floor(mtx_reco*tosf);
mtx_os = mtx_os +sw - mod(mtx_os,sw);
osf = (mtx_os/mtx_reco);


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

griddingrecon = sqrt(sum(abs(Atb).^2,4));
figure(3);imagesc(abs(griddingrecon(:,:,mtx_reco/2)))

%%
psf = FT'*dcf2;
H = (fftn(fftshift(psf)));

clear FT;
g = gpuDevice();
reset(g);
%%
opts.rho_r   = 2;
opts.beta    = [1 1 1];
opts.print   = true;
opts.alpha   = 0.7;
opts.method  = 'l2';
opts.tol = 1e-8;
opts.max_itr = 30;
Hnew = H./abs(H(1,1,1));

mu    = 2e3;
opts.rho_r = 1000;
opts.print=false;

out = zeros([mtx_reco*[1,1,1],nvCh],class(dd_2));
for i=1:nvCh,
     recon = deconvtvl2((Atb(:,:,:,i)),(Hnew),mu,opts);
     out(:,:,:,i) = gather(recon.f);
end

tvrecon = sqrt(sum(abs(out).^2,4));
figure(2);imagesc(abs(tvrecon(:,:,mtx_reco/2)));

fname = ['/Shared/lss_jcb/jcb/scan21Feb/P76800_',num2str(mtx_reco),'_',num2str(mu)];
save(fname,'tvrecon','griddingrecon');