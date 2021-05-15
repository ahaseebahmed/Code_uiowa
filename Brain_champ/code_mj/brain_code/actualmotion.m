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
addpath(genpath('/Users/ahhmed/Codes/ScannerCodes/'))

%d = './../17Jan20/P49152_rfs2.7';
%d = '/Users/jcb/abdul_brain/31Jan20/P00000_0.5Res_motion.7'
%wf_name = '/Users/jcb/abdul_brain/31Jan20/radial3D_1H_fov200_mtx400_intlv63200_kdt8_gmax33_smax118_dur5p2_fsca.mat'

d = '/Shared/lss_jcb/abdul/scan21Feb/P76800.7'
wf_name = '/Shared/lss_jcb/abdul/scan21Feb/radial3D_1H_fov200_mtx800_intlv147000_kdt6_gmax16_smax119_dur3p2_coca'

%d = '/Shared/lss_jcb/abdul/scan21Feb/P78336.7'
%wf_name = '/Shared/lss_jcb/abdul/scan21Feb/radial3D_1H_fov200_mtx800_intlv147000_kdt6_gmax16_smax119_dur3p2_coca.mat'
%d = './../31Jan20/P68608_woMotion.7';
%d = './../17Jan20/20200113_141044_P32768.7'; 
%d = './17Jan20/P52224.7'
%wf_name = './../31Jan20/radial3D_1H_fov200_mtx400_intlv63200_kdt8_gmax33_smax118_dur5p2_fsca.mat';
%wf_name = './../17Jan20/radial3D_1H_fov224_mtx448_intlv101460_kdt4_gmax17_smax118_dur1p6_coca.mat';
[dd,k,dcf,mtx_acq,nangles,fov,phi,theta]  = readradial(d,[],wf_name);

nsamples = length(dcf)/nangles;

nCh = size(dd,1);
dd = reshape(dd,nCh,nsamples,nangles);
k = reshape(k,nsamples,nangles,3);
dcf = reshape(dcf,nsamples,nangles);

load(wf_name);
Nperiods = 350;
indices = find((phi==0) & (theta==0));
period = indices(5)-indices(4);
s = [];
for i=0:2
    q = ifft(squeeze(dd(:,:,period-i:period:period*Nperiods)),[],2);
    q = reshape(permute(q,[2,1,3]),nCh*nsamples,size(q,3));
    s = [s;q];
end

nBasis = 5;
[~, ~, L] = estimateLapKernelLR(s(:,2:end), 1, 1);
[~,Sbasis,V]=svd(L);
Vbasis=V(:,end-nBasis+1:end);
Sbasis=Sbasis(end-nBasis+1:end,end-nBasis+1:end);

%%
ind = ones(size(dd,2),size(dd,3));
for i=period:period:nangles,
    dcf(:,i-2:i)=0;
    ind(:,i-2:i)=0;
end
 
%
%Channel combination

tmp = reshape(dd,nCh,nsamples*nangles);
Rs = gather(real(tmp*tmp'));

  [Vchannels,s] = eig(Rs);
  s = diag(s);[s,i] = sort(s,'descend');
  Vchannels = Vchannels(:,i);
  s=s./sum(s);s = cumsum(s);
  nvCh = min(6,nCh);%min(find(s>0.5));
  Vchannels = Vchannels(:,1:nvCh);
  dd_v = reshape(Vchannels'*tmp,[nvCh,nsamples,nangles]);
  dcf = reshape(dcf,[nsamples,nangles]);

  clear tmp;
%%
mtx_lr = 100;
load(wf_name);

nangles = Nperiods*period;
clear params;
plr.phi = phi(:,period+1:nangles);
plr.theta = theta(:,period+1:nangles);
plr.nS = size(ks,2);
plr.dd = dd_v(:,:,period+1:nangles);
plr.dcf = dcf(:,period+1:nangles);
plr.ks = ks*1.2*mtx_acq/mtx_lr; 
plr.indices = squeeze(find(sum(abs(plr.ks).^2,3)<0.25));


plr.ks = plr.ks(:,plr.indices,:);
plr.nks = length(plr.ks);

plr.mtx_reco = mtx_lr;

tform = affine3d(eye(4));
plr = transformKspaceAndData(tform,plr); 
temp = bsxfun(@times,transpose((plr.dd)),(plr.dcf(:)));
tic;lr_recon = plr.FT'*temp;toc
lr_recon = sqrt(sum(abs(lr_recon).^2,4));

figure(1);imagesc(abs(lr_recon(:,:,mtx_lr/2)));
%%
mtx_hr = 250;
load(wf_name);

clear params;
phr.phi = phi(:,period+1:nangles);
phr.theta = theta(:,period+1:nangles);
phr.nS = size(ks,2);
phr.dd = dd_v(:,:,period+1:nangles);
phr.dcf = dcf(:,period+1:nangles);
phr.ks = ks*1.2*mtx_acq/mtx_hr; 
phr.indices = squeeze(find(sum(abs(phr.ks).^2,3)<0.25));


phr.ks = phr.ks(:,phr.indices,:);
phr.nks = length(phr.ks);

phr.mtx_reco = mtx_hr;

tform = affine3d(eye(4));
phr = transformKspaceAndData(tform,phr); 
temp = bsxfun(@times,transpose((phr.dd)),(phr.dcf(:)));
tic;hr_recon = phr.FT'*temp;toc
hr_recon = sqrt(sum(abs(hr_recon).^2,4));
figure(2);imagesc(abs(hr_recon(:,:,mtx_hr/2)),[0,2e6])

%%
% ind = [-floor(plr.mtx_reco/2):floor(plr.mtx_reco/2)-1];
% [x,y,z] = meshgrid(ind,ind,ind);
% mask = x.^2+y.^2+z.^2 < 0.9*plr.mtx_reco^2/4;
% initial_sos = sqrt(sum(abs(initial_recon).^2,4)).*mask;
% figure(1); imagesc(abs(initial_sos(:,:,plr.mtx_reco/2)));title('initial sos');colormap(gray)

%%
csm = ones(plr.mtx_reco*[1,1,1]);
kdata = reshape(plr.dd(1,:),[plr.nks*period,size(Vbasis,1)]);
Atb = Atb_UV(plr.FT,kdata,Vbasis,csm,plr.mtx_reco,true);

M = @(x) AtA_UV(plr.FT,x,Vbasis,csm,plr.mtx_reco,plr.nks*period) + 0.1*x;
coeffs = pcg(M,Atb(:),1e-8,250);

coeffs = reshape(coeffs,plr.mtx_reco^3,size(Vbasis,2));
%%
angles = zeros(size(Vbasis,1),6);
img1 = coeffs*Vbasis(1,:)';
img1 = reshape(img1,mtx_lr*[1,1,1]);
 
for i=2:size(Vbasis,1)
    img2 = coeffs*Vbasis(i,:)';
    img2 = reshape(img2,mtx_lr*[1,1,1]);
    
       
    imagesc(abs(img2(:,:,50))); title(num2str(i));pause(0.05)
    [t3,tMatlab] = estRigid(abs(img2),abs(img1));
  
    deformParams = [rotm2eul(t3.T(1:3,1:3))'*180/pi,t3.T(4,1:3)*phr.mtx_reco/plr.mtx_reco];
    angles(i,:) =    deformParams;

end
%%
phase = ones(size(k,1),1);
k = phr.k;
for i=2:size(Vbasis,1)-1
    
    Thigh = i*period*phr.nks;
    Thighend = Thigh+period*phr.nks-1;
   
    deformParams =     angles(i,:);
    [k(Thigh:Thighend,:),phase(Thigh:Thighend)] = rotateKspaceAndData(k(Thigh:Thighend,:),deformParams);
end
phase = [phase;1];
pc = updateKspaceTraj(phr,k);
temp = bsxfun(@times,transpose(pc.dd),conj(phase));
temp = bsxfun(@times,temp,pc.dcf(:));
tic;coilImagesC = pc.FT'*temp;toc
temp = bsxfun(@times,temp,pc.dcf(:));
reconCorrected = sqrt(sum(abs(coilImagesC).^2,4));      

temp = bsxfun(@times,transpose(phr.dd),phr.dcf(:));
tic;coilImagesUc = phr.FT'*temp;toc
reconUnCorrected = sqrt(sum(abs(coilImagesUc).^2,4));      

figure(6);imagesc(abs(reconCorrected(:,:,125)),[0,2e6]);title('Corrected');
figure(7);imagesc(abs(reconUnCorrected(:,:,125)),[0,2e6]);title('UnCorrected');

%% Try low resolution reconstruction
opts.rho_r   = 2;
opts.beta    = [1 1 1];
opts.print   = true;
opts.alpha   = 0.7;
opts.method  = 'l2';
opts.tol = 1e-8;
opts.max_itr = 30;

mu    = 1e4;
opts.rho_r = 1000;
%opts.print=false;

%%
psf = pc.FT'*pc.dcf(:);
H = (fftn(fftshift(psf)));
Hnew = H./abs(H(1,1,1));

out = zeros([pc.mtx_reco*[1,1,1],nvCh]);
for i=1:nvCh,
     recon = deconvtvl2(gpuArray(coilImagesC(:,:,:,i)),gpuArray(Hnew),mu,opts);
     out(:,:,:,i) = gather(recon.f);
end

corrrected = sqrt(sum(abs(out).^2,4));
figure(6);imagesc(abs(corrrected(:,:,pc.mtx_reco/2)));

%%

psf = phr.FT'*pc.dcf(:);
H = (fftn(fftshift(psf)));
Hnew = H./abs(H(1,1,1));

out = zeros([phr.mtx_reco*[1,1,1],nvCh]);
for i=1:nvCh,
     recon = deconvtvl2(gpuArray(coilImagesUc(:,:,:,i)),gpuArray(Hnew),mu,opts);
     out(:,:,:,i) = gather(recon.f);
end

uncorrrected = sqrt(sum(abs(out).^2,4));
figure(7);imagesc(abs(uncorrrected(:,:,pc.mtx_reco/2)));