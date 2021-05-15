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

fname = ['/Shared/lss_jcb/abdul/recon_3dradial/P82432_400_4000.mat'];
wf_name = '/Shared/lss_jcb/abdul/scan27Feb/radial3D_1H_fov200_mtx600_intlv147000_kdt8_gmax12_smax117_dur3p2_coca'

load(fname);
load(wf_name);
indices = find((phi==0) & (theta==0));
period = indices(5)-indices(4);

ktrue = giveTraj(ks, phi, theta);
dcf = repmat(dcfs,[1,length(phi)]);
dcf(:,period:period:end) = 0;

dcf(:,period-1:period:end) = 0;
dcf(:,period-2:period:end) = 0;

dcf = dcf(:);

osf = 1.25; % oversampling: 1.5 1.25
wg = 3; % kernel width: 5 7
sw = 8; % parallel sectors' width: 12 16
mtx_reco = 400;

FT = gpuNUFFT(transpose(ktrue),ones(size(dcf(:))),osf,wg,sw,[1 1 1]*mtx_reco,[],true);

[cmap,mask] = giveCSM(tvrecon,4,12,0.1);
tvrecon = tvrecon.*mask;

kdata = FT*tvrecon;
%test = FT'*(kdata.*dcf(:));

%%
nks = size(ks,2);
nspokes = length(phi);
nperiods = nspokes/period;

% anglestrue = [[0,0,0,0,7,7];
%               [4,3,0,7,0,0];
%               [0,9,0,0,-5,0];
%               [0,0,9,0,7,0]];
          
          
anglestrue = zeros(nperiods,6);
anglestrue(87:168,1) = 0;
anglestrue(87:168,2) = 0;
anglestrue(87:168,3) = 0;
anglestrue(87:168,4) = 7; 
anglestrue(87:168,5) = -2;
anglestrue(87:168,6) = 3;

anglestrue(169:230,1) = -linspace(3,2,230-169+1);
anglestrue(169:230,2) = linspace(1,2,230-169+1);
anglestrue(169:230,3) = 0;
anglestrue(169:230,4) = linspace(0,3,230-169+1);
anglestrue(169:230,5) = linspace(-3,1,230-169+1);
anglestrue(169:230,6) = 0;

anglestrue(231:320,1) = 0.2;
anglestrue(231:320,2) = -1;
anglestrue(231:320,3) = 0;
anglestrue(231:320,4) = 3;
anglestrue(231:320,5) = 2;
anglestrue(231:320,6) = 0;


anglestrue(321:420,1) = 0;
anglestrue(321:420,2) = 0;
anglestrue(321:420,3) = -2;
anglestrue(321:420,4) = -2;
anglestrue(321:420,5) = -2;
anglestrue(321:420,6) = -3;

anglestrue = conv2(anglestrue,[1,1,1,1,1,1,1]'/7,'same');

k= ktrue;
k = reshape(k,[nks,period,nperiods,3]);
phase = ones(nks,period,nperiods);
for i=1:nperiods
    [ktemp,ptemp] = rotateKspaceAndData(k(:,:,i,:),anglestrue(i,:));
    k(:,:,i,:) = ktemp;
    phase(:,:,i) = ptemp;
end
k = reshape(k,nks*period*nperiods,3);
phase = reshape(phase,nks*period*nperiods,1);

% k = reshape(k,[nks,period,nperiods,3]);
% phase = ones(nks,period,nperiods);
% for i=1:length(sP)
%     [ktemp,ptemp] = rotateKspaceAndData(k(:,:,sP(i):eP(i),:),anglestrue(i,:));
%     k(:,:,sP(i):eP(i),:) = ktemp;
%     phase(:,:,sP(i):eP(i)) = ptemp;
% end
% k = reshape(k,nks*period*nperiods,3);
% phase = reshape(phase,nks*period*nperiods,1);
% 
FTperturbed = gpuNUFFT(transpose(k),ones(size(dcf(:))),osf,wg,sw,[1 1 1]*mtx_reco,[],true);
kdata = FTperturbed*tvrecon;
test = FTperturbed'*(kdata.*dcf(:));
kdata = kdata.*(phase);
%
kdata = reshape(kdata,nks,period,nperiods);

%% Motion due to ratotation and translation
test = FT'*(kdata(:).*dcf(:));

%% Reconstruction and motion estimation

display('Reconstructing the time series')
L1 = circulant([1,-1,zeros(1,nperiods-2)]);
Lnn = L1'*L1/2;
navs = kdata(:,end-2:end,:);
navs = fft(navs,[],1);
navs = reshape(navs,[nks*3,nperiods]);
nBasis = 30;
[~, ~, L] = estimateLapKernelLR(navs, 1, 1);
[~,Sbasis,V]=svd(L+0.0*Lnn);
Vbasis=V(:,end-nBasis+1:end);
Sbasis=Sbasis(end-nBasis+1:end,end-nBasis+1:end);

% for i=1:6
%     subplot(2,3,i);
%     plot(Vbasis(:,end-i));
%     hold on;
%     plot(Vbasis1(:,end-i));
% end
% 
for i=1:6
    subplot(2,3,i);
    hold on;
    if i<4
        plot(angles(:,i))
    else
       plot(-angles(:,i))
    end
    
    plot(anglestrue(:,i))
end

mtx_small = 200;

ksSmall = ks*mtx_reco/mtx_small; 
indices = squeeze(find(sum(abs(ksSmall).^2,3)<0.5));

nks = length(ks);
dcfsmall = reshape(dcf,[nks,nspokes]);
ksmall = reshape(ktrue,[nks,nspokes,3]);
ksmall = ksmall*mtx_reco/mtx_small;
kdatasmall = reshape(kdata,[nks,nspokes]);

dcfsmall = dcfsmall(indices,:);
ksmall = ksmall(indices,:,:);
kdatasmall = kdatasmall(indices,:);

nksSmall = length(indices);
ksmall = reshape(ksmall,nksSmall*nspokes,3);
dcfsmall = reshape(dcfsmall,nksSmall*nspokes,1);
kdatasmall = reshape(kdatasmall,[nksSmall*nspokes,1]);

FT = gpuNUFFT(transpose(ksmall),ones(size(dcfsmall(:))),osf,wg,sw,[1 1 1]*mtx_small,[],true);
test = FT'*(kdatasmall(:).*dcfsmall(:));
imagesc(abs(test(:,:,mtx_small/2)));


csm = ones(mtx_small*[1,1,1]);
kdatasmall = reshape(kdatasmall,[nksSmall*period,size(Vbasis,1)]);
%Atb = Atb_UV(FT,kdata,Vbasis,csm,mtx_small,true);
Atb = Atb_UV_T1(FT,kdatasmall(:)',Vbasis,dcfsmall,mtx_small);

%%%%%
%FT = gpuNUFFT(transpose(ksmall),((dcfsmall(:))),osf,wg,sw,[1 1 1]*mtx_small,[],true);

%Atb = Atb_UV( FT, kdatasmall(:)', Vbasis, csm, mtx_small, true);
%Reg = @(x) reshape( reshape( x, [N*N*N, nbasis]) * SBasis, [N*N*N*nbasis, 1]);
%AtA = @(x) AtA_UV( FT, x, V, csm, N, nSamplesPerFrame, imageDim) + Reg( x);
%%%%%

Reg = @(x) reshape( reshape( x, [mtx_small^3, nBasis]) * Sbasis, [(mtx_small^3)*nBasis, 1]);
M = @(x) AtA_UV(FT,x,Vbasis,csm,mtx_small,nksSmall*period) + 0.3*Reg(x);
coeffs = pcg(M,Atb(:),1e-8,4,[],[],Atb(:));

coeffs = reshape(Atb,mtx_small^3,size(Vbasis,2));

img1 = coeffs*Vbasis(1,:)';
img1 = reshape(img1,mtx_small*[1,1,1]);
display('Reconstructed the time series')

%% Motion estimation and compensation

display('Motion estimation')

angles = zeros(nperiods,6);
for ii=2:nperiods,
    img2 = coeffs*Vbasis(ii,:)';
    img2 = reshape(img2,mtx_small*[1,1,1]);
   % imagesc(abs(img2(:,:,35))); title(num2str(i));
    %pause(1);
    [t3,tMatlab] = estRigid(abs(img1),abs(img2));
    deformParams = [rotm2eul(t3.T(1:3,1:3))'*180/pi,t3.T(4,1:3)*mtx_reco/mtx_small];
    angles(ii,:) =    deformParams;
end
k = ktrue;

k = reshape(k,[nks,period,nperiods,3]);
phase = ones(nks,period,nperiods);
for i=1:nperiods
    [ktemp,ptemp] = rotateKspaceAndData(k(:,:,i,:),angles(i,:));
    k(:,:,i,:) = ktemp;
    phase(:,:,i) = ptemp;
end
k = reshape(k,nks*period*nperiods,3);
phase = reshape(phase,nks*period*nperiods,1);

FT = gpuNUFFT(transpose(k),ones(size(dcf(:))),osf,wg,sw,[1 1 1]*mtx_reco,[],true);
test = FT'*(kdata(:).*dcf(:).*(phase(:)));
imagesc(abs(test(:,:,200)),[0,1e7]); title(num2str(i));
psf = FT'*dcf(:);
H = (fftn(fftshift(psf)));

opts.rho_r   = 2;
opts.beta    = [1 1 1];
opts.print   = true;
opts.alpha   = 0.7;
opts.method  = 'l2';
opts.tol = 1e-8;
opts.max_itr = 60;
Hnew = H./abs(H(1,1,1));

mu    = 2;
opts.rho_r = 1e10;
opts.print=true;
recon = deconvtvl2((test),(H),mu,opts);

FT = gpuNUFFT(transpose(ktrue),ones(size(dcf(:))),osf,wg,sw,[1 1 1]*mtx_reco,[],true);
testcorrupted = FT'*(kdata(:).*dcf(:));
figure(2);imagesc(abs(testcorrupted(:,:,200)),[0,1e7]); title(num2str(i));
psf1 = FT'*dcf(:);
H1= (fftn(fftshift(psf1)));
reconcorrupted = deconvtvl2((testcorrupted),(H),mu,opts);

%%

for i=2:nperiods,
    img2 = coeffs*Vbasis(i,:)';
    img2 = reshape(img2,mtx_small*[1,1,1]);
    figure(1);imshowpair(abs(img2(:,:,50)),abs(img1(:,:,50)),'Scaling','Joint');
    title(num2str(i))
    pause(0.01);
end

% i=280;img2 = coeffs*Vbasis(i,:)';  img2 = reshape(img2,mtx_small*[1,1,1]);
% figure(1);imshowpair(abs(img2(:,:,50)),abs(img1(:,:,50)),'Scaling','Joint');

