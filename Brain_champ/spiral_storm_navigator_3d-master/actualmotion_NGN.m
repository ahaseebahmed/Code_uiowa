clear all
%cd '/Users/jcb/abdul_brain/code'
addpath(genpath('/Users/ahhmed/Codes/Data/Brain_champ/spiral_storm_navigator_3d-master/'))
addpath(genpath('/Users/ahhmed/Codes/Data/Brain_champ/code_mj/brain_code'))
%d = '/Shared/lss_jcb/abdul/scan21Feb/P76800.7'
%wf_name = '/Shared/lss_jcb/abdul/scan21Feb/radial3D_1H_fov200_mtx800_intlv147000_kdt6_gmax16_smax119_dur3p2_coca'

%d = '/Shared/lss_jcb/abdul/scan21Feb/P78336.7'
%wf_name = '/Shared/lss_jcb/abdul/scan21Feb/radial3D_1H_fov200_mtx800_intlv147000_kdt6_gmax16_smax119_dur3p2_coca.mat'
%d = './../31Jan20/P68608_woMotion.7';
%d = './../17Jan20/20200113_141044_P32768.7'; 
%d = './17Jan20/P52224.7'
%wf_name = './../31Jan20/radial3D_1H_fov200_mtx400_intlv63200_kdt8_gmax33_smax118_dur5p2_fsca.mat';
%wf_name = './../17Jan20/radial3D_1H_fov224_mtx448_intlv101460_kdt4_gmax17_smax118_dur1p6_coca.mat';
%load('/Shared/lss_jcb/abdul/forSSN_3d_P63488.mat'); with motion
load('/Shared/lss_jcb/abdul/forSSN_3d_P12288.mat');%silent wiht motion low res.
load('/Shared/lss_jcb/abdul/ngnp/forSSN_3d_P62464.mat');%T1 silent
load('/Shared/lss_jcb/abdul/ngnp/forSSN_3d_P61952.mat');%silent
load('/Shared/lss_jcb/abdul/ngnp/forSSN_3d_P64512.mat');%T2silent with motion
load('/Shared/lss_jcb/abdul/ngnp/forSSN_3d_P64000.mat');%T1silent with motion
load('/Shared/lss_jcb/abdul/ngnp/forSSN_3d_P62976.mat');%T2silent

%load('/Users/ahhmed/Codes/Data/Brain_champ/spiral_storm_navigator_3d-master/simulatedData/030121/forSSN_3d_silent_grad_25.mat')
%load('/Users/ahhmed/Codes/Data/Brain_champ/spiral_storm_navigator_3d-master/simulatedData/030121/silent_grad_25_storm_poof_n64_rot_motion_20210226T211922/forSSN_3d_silent_grad_25.mat')
%load('/Users/ahhmed/Codes/Data/Brain_champ/spiral_storm_navigator_3d-master/simulatedData/031621/silent_grad_25_recon_96_n32_rot_motion_64_n112_20210314T155521/forSSN_3d_silent_grad_25.mat');


%load('/Users/ahhmed/Codes/Data/Brain_champ/spiral_storm_navigator_3d-master/simulatedDataMC/210405_4ch_for_abdul/silent_grad_25_recon_4ch_csm_rescube_20210405T230620/forSSN_3d_silent_grad_25.mat');
load('/Users/ahhmed/Codes/Data/Brain_champ/spiral_storm_navigator_3d-master/simulatedData/210427_1ch_full_sample_for_abdul/silent_grad_25_recon_os2p0_kw4p0_20210427T174127/forSSN_3d_silent_grad_25.mat');

%%
% mov_ker=2;
% m_avg=ones(mov_ker,1)*1/mov_ker;
mtx_hr=256;
nks=128;period=192;nperiods=64;
osf=1.5;wg=3;sw=8;
korig=permute(ktraj,[2,3,4,1]);
korig = reshape(korig,[nks*period*nperiods,3]);
%kdata=kdata./max(abs(kdata(:)));
% kdata_filt=[kdata(1,:,:);kdata;kdata(end,:,:)];
% for i=1: size(kdata_filt,2)
%     for j=1:size(kdata_filt,3)
%         kdata_filt(:,i,j)=filter(m_avg,1,kdata_filt(:,i,j));
%     end
% end
% kdatan=kdata_filt(2:end-1,:,:);

korig=0.5*korig;
FT = gpuNUFFT(transpose(korig),ones(size(dcf(:))),osf,wg,sw,[1 1 1]*mtx_hr,[],true);
mo_crpt0 = FT'*(kdata(:).*dcf(:).^0);
kdata1=reshape(kdata,[nks*period*nperiods,4]);
mo_crpt0 = FT'*(kdata1.*dcf(:).^1);




f_mo_crpt0=fftshift(fftn(mo_crpt0));
%f_mo_crpt0=mo_crpt0;
f_mo_crpt0=f_mo_crpt0./max(abs(f_mo_crpt0(:)));


figure(1);colormap gray;
subplot(131);
imagesc(abs(f_mo_crpt0(:,:,mtx_hr/2)));
axis off; 
subplot(132);
imagesc((squeeze(abs(f_mo_crpt0(:,mtx_hr/2,:)))));
axis off;
subplot(133);
imagesc((squeeze(abs(f_mo_crpt0(mtx_hr/2,:,:))))); 
axis off;

%mo_crpt0=mo_crpt0./max(abs(mo_crpt0(:)));
Reg = @(x) reshape(l2_norm_TV(reshape(x,[mtx_hr,mtx_hr,mtx_hr])),[mtx_hr^3,1]);
ATA = @(x) (reshape(FT'*(FT*reshape(x,[mtx_hr,mtx_hr,mtx_hr]).*dcf(:).^0),[mtx_hr^3,1]))+0.5*Reg(x);
tic();[x0,~,~,~,res]=pcg(ATA,mo_crpt0(:),1e-10,100);toc();

A=@(x) FT*x;
At=@(x) FT'*x;

tic();[X] = OpTV(kdata(:),A,At,0.0);toc();


mo_crpt=reshape(x0,[mtx_hr,mtx_hr,mtx_hr]);

mo_crpt=mo_crpt./max(abs(mo_crpt(:)));
f_mo_crpt0=mo_crpt;
f_mo_crpt0=fftshift(fftn(mo_crpt));

figure(6);colormap gray;
subplot(131);
imagesc(abs(mo_crpt(:,:,mtx_hr/2)));
axis off; 
subplot(132);
imagesc((squeeze(abs(mo_crpt(:,mtx_hr/2,:)))));
axis off;
subplot(133);
imagesc((squeeze(abs(mo_crpt(mtx_hr/2,:,:))))); 
axis off;

f_mo_crpt0=f_mo_crpt0./max(abs(f_mo_crpt0(:)));



gwin = giveMask(mo_crpt,13,0);
gwin=1-gwin;
gwin = reshape(gwin,[mtx_hr*mtx_hr*mtx_hr,1]);
params.gwin=gwin;

[img1, x, Svalues, V, params, flags] = spiral_storm_navigator_3d_011221(kdata,knav,ktraj,dcf);
imgu=img;
img=imgR;
img=img1;
img=img./max(abs(img(:)));
v=VideoWriter('axil2.avi');
v.FrameRate=10
open(v)
for i=3:size(img,4)
writeVideo(v,imadjust(squeeze(abs(img(:,:,size(img,3)/2,i)))));
end
close(v)

v=VideoWriter('corr2.avi');
v.FrameRate=10
open(v)
for i=3:size(img,4)
writeVideo(v,imadjust(squeeze(abs(img(:,size(img,2)/2,:,i)))));
end
close(v)


v=VideoWriter('sagi2.avi');
v.FrameRate=10
open(v)
for i=3:size(img,4)
writeVideo(v,imadjust(squeeze(abs(img(size(img,1)/2,:,:,i)))));
end
close(v)


%%
% mtx_hr = 250;
% load(wf_name);
% nangles = Nperiods*period;
% 
% clear params;
% phr.phi = phi(:,period+1:nangles);
% phr.theta = theta(:,period+1:nangles);
% phr.nS = size(ks,2);
% phr.dd = dd_v(:,:,period+1:nangles);
% phr.dcf = dcf(:,period+1:nangles);
% phr.ks = ks*1.2*mtx_acq/mtx_hr; 
% phr.indices = squeeze(find(sum(abs(phr.ks).^2,3)<0.5));
% 
% 
% phr.ks = phr.ks(:,phr.indices,:);
% phr.nks = length(phr.ks);
% 
% phr.mtx_reco = mtx_hr;
% 
% tform = affine3d(eye(4));
% phr = transformKspaceAndData(tform,phr); 
% temp = bsxfun(@times,transpose((phr.dd)),(phr.dcf(:)));
% tic;hr_recon = phr.FT'*temp;toc
% hr_recon = sqrt(sum(abs(hr_recon).^2,4));
% figure(2);imagesc(abs(hr_recon(:,:,mtx_hr/2)),[0,2e6])
% figure(3);imagesc(abs(hr_recon(:,:,mtx_hr/2)))

%%
% ind = [-floor(plr.mtx_reco/2):floor(plr.mtx_reco/2)-1];
% [x,y,z] = meshgrid(ind,ind,ind);
% mask = x.^2+y.^2+z.^2 < 0.9*plr.mtx_reco^2/4;
% initial_sos = sqrt(sum(abs(initial_recon).^2,4)).*mask;
% figure(1); imagesc(abs(initial_sos(:,:,plr.mtx_reco/2)));title('initial sos');colormap(gray)

%%
tic();
mtx_hr=512;
mtx_lr=256;
x=reshape(x,[mtx_lr^3,size(V,2)]);
angles = zeros(size(V,1),6);
img1 = x*V(6,:)';
img1 = reshape(img1,mtx_lr*[1,1,1]);
 
for i=1:size(V,1)
%     img1 = x*V(i,:)';
%     img1 = reshape(img1,mtx_lr*[1,1,1]);
    img2 = x*V(i,:)';
    img2 = reshape(img2,mtx_lr*[1,1,1]);
    
       
    %imagesc(abs(img2(:,:,50))); title(num2str(i));pause(0.05)
    [t3,tMatlab,imgR(:,:,:,i)] = estRigid(abs(img2(mtx_lr/2-mtx_lr/8:mtx_lr/2+mtx_lr/8,mtx_lr/2-mtx_lr/8:mtx_lr/2+mtx_lr/8,mtx_lr/2-mtx_lr/8:mtx_lr/2+mtx_lr/8)),...
        abs(img1(mtx_lr/2-mtx_lr/8:mtx_lr/2+mtx_lr/8,mtx_lr/2-mtx_lr/8:mtx_lr/2+mtx_lr/8,mtx_lr/2-mtx_lr/8:mtx_lr/2+mtx_lr/8)));
  
    deformParams = [rotm2eul(t3.T(1:3,1:3))'*180/pi,t3.T(4,1:3)*mtx_hr/mtx_lr];
    angles(i,:) =    deformParams;

end
toc();
%%

%84:96,112:127,189:201,228:236
idx=[189:201,228:236];
idx=[1:4,13:20,29:36,45:52,62:68,115:130,139:147,155:162,239:242,247:251];
%nks=128;period=192;
nperiods=size(V,1);
osf=1.5;wg=3;sw=8;
k=permute(ktraj,[2,3,4,1]);
k(:,:,idx,:)=[];
nperiods=size(k,3)
k = reshape(k,[nks,period,nperiods,3]);
phase = ones(nks,period,nperiods);
for i=1:nperiods
    [ktemp,ptemp] = rotateKspaceAndData(k(:,:,i,:),angles(i,:));
    k(:,:,i,:) = ktemp;
    phase(:,:,i) = ptemp;
end
k = reshape(k,nks*period*nperiods,3);
phase = reshape(phase,nks*period*nperiods,1);

k=0.5*k;
kdata1=kdata;
dcf1=dcf;
dcf1(:,:,idx)=[];
kdata1(:,:,idx)=[];
FT = gpuNUFFT(transpose(k),ones(size(dcf1(:))),osf,wg,sw,[1 1 1]*mtx_hr,[],true);
test = FT'*(kdata(:).*(phase(:)).*dcf(:).^1);
test = FT'*(kdata1(:).*(phase(:)).*dcf1(:).^1);

Reg = @(x) reshape(l2_norm_TV(reshape(x,[mtx_hr,mtx_hr,mtx_hr])),[mtx_hr^3,1]);
ATA = @(x) (reshape(FT'*((FT*reshape(x,[mtx_hr,mtx_hr,mtx_hr]).*dcf1(:).^1)),[mtx_hr^3,1]))+0.005*Reg(x);
tic();x1=pcg(ATA,test(:),1e-5,20);toc();
test=reshape(x1,[mtx_hr,mtx_hr,mtx_hr]);
figure(3);colormap gray;
subplot(131);
imagesc(abs(test(:,:,mtx_hr/2)));
axis off; 
subplot(132);
imagesc((squeeze(abs(test(:,mtx_hr/2,:)))));
axis off; 
subplot(133);
imagesc((squeeze(abs(test(mtx_hr/2,:,:))))); 
axis off; 

%%-----------------knav motion estimation-------------
period_nav=64;
kn = reshape(ktraj_nav,[nks,period_nav,nperiods,3]);
phase_n = ones(nks,period_nav,nperiods);
for i=1:nperiods
    [ktemp,ptemp] = rotateKspaceAndData(kn(:,:,i,:),angles(i,:));
    kn(:,:,i,:) = ktemp;
    phase_n(:,:,i) = ptemp;
end
kn = reshape(kn,nks,period_nav,nperiods,3);
phase_n = reshape(phase_n,nks*period_nav*nperiods,1);

k=reshape(k,[nks,period,nperiods,3]);
ktraj1=permute(k,[4,1,2,3]);
kdata1=kdata(:).*(phase(:));
kdata1=reshape(kdata1,[nks,period,nperiods]);
knav1=knav(:).*(phase_n(:));
knav1=reshape(knav1,[nks,period_nav,nperiods]);
ktraj_nav=reshape(ktraj_nav,[nks*period_nav*nperiods,3]);
kn=reshape(kn,[nks*period_nav*nperiods,3]);
knav1=reshape(knav1,[nks*period_nav*nperiods,1]);
knav2=griddata(ktraj_nav(:,1),ktraj_nav(:,2),ktraj_nav(:,3),knav1(:),kn(:,1),kn(:,2),kn(:,3));

[img1, x, Svalues, V, params, flags] = spiral_storm_navigator_3d_011221(kdata1,knav,ktraj1,dcf);


% phase = ones(size(k,1),1);
% k = phr.k;
% for i=2:size(Vbasis,1)-1
%     
%     Thigh = i*period*phr.nks;
%     Thighend = Thigh+period*phr.nks-1;
%    
%     deformParams =     angles(i,:);
%     [k(Thigh:Thighend,:),phase(Thigh:Thighend)] = rotateKspaceAndData(k(Thigh:Thighend,:),deformParams);
% end
% phase = [phase;1];
% pc = updateKspaceTraj(phr,k);
% temp = bsxfun(@times,transpose(pc.dd),(phase));
% temp = bsxfun(@times,temp,pc.dcf(:));
% tic;coilImagesC = pc.FT'*temp;toc
% temp = bsxfun(@times,temp,pc.dcf(:));
% reconCorrected = sqrt(sum(abs(coilImagesC).^2,4));      
% 
% temp = bsxfun(@times,transpose(phr.dd),phr.dcf(:));
% tic;coilImagesUc = phr.FT'*temp;toc
% reconUnCorrected = sqrt(sum(abs(coilImagesUc).^2,4));      
% 
% figure(6);imagesc(abs(reconCorrected(:,:,125)),[0,2e6]);title('Corrected');
% figure(7);imagesc(abs(reconUnCorrected(:,:,125)),[0,2e6]);title('UnCorrected');

%% Try low resolution reconstruction
opts.rho_r   = 2;
opts.beta    = [1 1 1];
opts.print   = true;
opts.alpha   = 0.7;
opts.method  = 'l2';
opts.tol = 1e-8;
opts.max_itr = 200;

mu    = 1e3;
opts.rho_r = 1000;
%opts.print=false;

%%
nvCh=1;
coilImagesC=test;
psf = FT'*dcf(:);
H = (fftn(fftshift(psf)));
Hnew = H./abs(H(1,1,1));

out = zeros([mtx_hr*[1,1,1],nvCh]);
for i=1:nvCh,
     recon = deconvtvl2(gpuArray(coilImagesC(:,:,:,i)),gpuArray(Hnew),mu,opts);
     out(:,:,:,i) = gather(recon.f);
end

corrrected = sqrt(sum(abs(out).^2,4));
figure(6);imagesc(abs(corrrected(:,:,mtx_hr/2)));

figure(3);colormap gray;
subplot(131);
imagesc(abs(corrrected(:,:,64))); 
subplot(132);
imagesc(rot90(squeeze(abs(corrrected(:,64,:))),1)); 
subplot(133);
imagesc(rot90(squeeze(abs(corrrected(64,:,:))),1)); 

%%
nvCh=1;
psf = FT'*pc.dcf(:);
H = (fftn(fftshift(psf)));
Hnew = H./abs(H(1,1,1));

out = zeros([mtx_hr*[1,1,1],nvCh]);
for i=1:nvCh,
     recon = deconvtvl2(gpuArray(coilImagesUc(:,:,:,i)),gpuArray(Hnew),mu,opts);
     out(:,:,:,i) = gather(recon.f);
end

uncorrrected = sqrt(sum(abs(out).^2,4));
figure(7);imagesc(abs(uncorrrected(:,:,pc.mtx_reco/2)));