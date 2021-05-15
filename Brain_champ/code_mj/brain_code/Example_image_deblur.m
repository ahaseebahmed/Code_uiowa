clear all

addpath(genpath('/Users/jcb/medimgviewer'));
addpath(genpath('./../../gpuNUFFT'));
addpath(genpath('./../../Fessler-nufft'));
clc
addpath(genpath('code'));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Demo file for deconvtv
% Image deblurring
% 
% Stanley Chan
% University of California, San Diego
% 20 Jan, 2011
%
% Copyright 2011
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

mtx_reco = 128;
% Prepare images
f_orig  = phantom3d(mtx_reco);%im2double(imread('./data/building.jpg'));
%f_orig = zeros(mtx_reco*[1,1,1]);
%f_orig(32:32+63,32:32+63,32:32+63) = test;
wf_name = '/Users/jcb/abdul_brain/scan17FebHuman/radial3D_1H_fov200_mtx800_intlv130800_kdt6_gmax33_smax119_dur8_fsca_fullspoke'
wf_name = '/Shared/lss_jcb/abdul/scan27Feb/radial3D_1H_fov200_mtx600_intlv147000_kdt8_gmax12_smax117_dur3p2_coca'

load(wf_name);

indices = find((phi==0) & (theta==0));
period = indices(5)-indices(4);

nsamples = length(ks);
nangles = length(phi);
k = giveTraj(ks, phi, theta);
k = reshape(k,nsamples,nangles,3); 
dcf = sqrt(sum(abs(k).^2,3));
dcf(:,1:period:end)=0;dcf(:,2:period:end)=0;dcf(:,3:period:end)=0;

nK = 40000;
k = reshape(k(:,1:nK,:),[nsamples*nK,3]); 
dcf = reshape(dcf(:,1:nK),[nsamples*nK,1]);

osf = 2; % oversampling: 1.5 1.25
wg = 3; % kernel width: 5 7
sw = 8; % parallel sectors' width: 12 16

FT = gpuNUFFT(transpose(k),ones(size(k,1),1),osf,wg,sw,[1 1 1]*mtx_reco,[],true);
b = FT*f_orig;

% init_recon = FT'*(b.*(dcf));
% 
% A = @(x) reshape(FT'*(FT*x),[mtx_reco^3,1]);
% Atb = FT'*b;
% 
% test = pcg(A,Atb(:),1e-4,50); test = reshape(test,mtx_reco*[1,1,1]);
% 
mtx_final = mtx_reco + 20;
FT1 = gpuNUFFT(transpose(k),ones(size(k,1),1),osf,wg,sw,[1 1 1]*(mtx_final),[],true);

psf = FT1'*dcf;
H = (fftn(fftshift(psf)));


Atb = FT1'*(b.*dcf);
%%
H = gpuArray(H);
B = @(x) reshape(ifftshift(ifftn(H.*fftn(fftshift(reshape(x,mtx_final*[1,1,1]))))),[mtx_final^3,1]);
test = pcg(B,gpuArray(Atb(:)),1e-4,100); test = reshape(test,mtx_final*[1,1,1]);
test = reshape(test,mtx_final*[1,1,1]);
figure(2);imagesc(abs(test(:,:,63)));
%%

% Setup parameters (for example)
opts.rho_r   = 2;
opts.beta    = [1 1 1];
opts.print   = true;
opts.alpha   = 0.7;
opts.method  = 'l2';
opts.tol = 1e-8;
opts.max_itr = 50;
% Main routine

tic
psf = FT1'*dcf;
H = (fftn(fftshift(psf)));
Hnew = H./abs(H(1,1,1));

%%
mu    = 1e6;
opts.rho_r = 1000;

out = deconvtvl2(gpuArray(Atb),gpuArray(Hnew),mu,opts);
%out = deconvtv(g, H, mu, opts);
figure(1);imagesc(abs(out.f(:,:,63)))
toc
%%
% Display results
figure(1);
imshow(abs(init_recon(:,:,64)),[]);
title('input');

figure(2);
imshow(abs(out.f(:,:,64)),[]);
title('output');