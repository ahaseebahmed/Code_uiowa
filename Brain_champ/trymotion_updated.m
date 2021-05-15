addpath(genpath('/Users/ahhmed/Codes/Data/Brain_champ/code_mj/brain_code'));
addpath(genpath('/Users/ahhmed/Codes/Data/Brain_champ/'));
d = '/Shared/lss_jcb/abdul/scan21Feb/P76800.7'
wf_name = '/Shared/lss_jcb/abdul/scan21Feb/radial3D_1H_fov200_mtx800_intlv147000_kdt6_gmax16_smax119_dur3p2_coca'


[dd,k,dcf,mtx_acq,nangles,fov,phi,theta]  = readradial(d,[],wf_name);

[U, V, kdata,dcf,ktraj_scaled]=spiral_storm_navigator_3d_updated();

k=permute(ktraj_scaled,[2,3,1]);

%% Motion estimation and compensation
mtx=128;
mtx_reco=128
nbasis=4;
nks=128;
period=12288/nks;

display('Motion estimation')
nperiods=size(V,1);
U=reshape(U,[mtx^3,nbasis]);
img1=reshape(U,[mtx^3,nbasis])*V(1,:)';
img1=reshape(img1,mtx*[1,1,1]);
angles = zeros(nperiods,6);
for ii=2:nperiods,
    img2 = U*V(ii,:)';
    img2 = reshape(img2,mtx*[1,1,1]);
   % imagesc(abs(img2(:,:,35))); title(num2str(i));
    %pause(1);
    [t3,tMatlab] = estRigid(abs(img1),abs(img2));
    deformParams = [rotm2eul(t3.T(1:3,1:3))'*180/pi,t3.T(4,1:3)*mtx_reco/mtx];
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

osf = 1.25; % oversampling: 1.5 1.25
wg = 3; % kernel width: 5 7
sw = 8;

ktrue=reshape(ktraj_scaled,[3,nks*period*nperiods]);
FT = gpuNUFFT(ktrue,ones(size(dcf(:))),osf,wg,sw,[1 1 1]*mtx_reco,[],true);
test = FT'*(kdata(:).*dcf(:));
imagesc(abs(test(:,:,mtx/2)));


FT = gpuNUFFT(transpose(k),ones(size(dcf(:))),osf,wg,sw,[1 1 1]*mtx_reco,[],true);
test = FT'*(kdata(:).*dcf(:).*(phase(:)));
imagesc(abs(test(:,:,mtx/2))); title(num2str(i));
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

