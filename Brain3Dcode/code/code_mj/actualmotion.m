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

%d = './../17Jan20/P49152_rfs2.7';
d = '/Users/jcb/abdul_brain/31Jan20/P00000_0.5Res_motion.7'
wf_name = '/Users/jcb/abdul_brain/31Jan20/radial3D_1H_fov200_mtx400_intlv63200_kdt8_gmax33_smax118_dur5p2_fsca.mat'
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
indices = find((phi==0) & (theta==0));
period = indices(5)-indices(4);

s = [];
for i=1:3
    q = ifft(squeeze(dd(:,:,i:period:end)),[],2);
    q = reshape(permute(q,[2,1,3]),nCh*nsamples,size(q,3));
    s = [s;q];
end

nBasis = 4;
[~, ~, L] = estimateLapKernelLR(s(:,2:end), 1, 1);
[~,Sbasis,V]=svd(L);
Vbasis=V(:,end-nBasis+1:end);
Sbasis=Sbasis(end-nBasis+1:end,end-nBasis+1:end);

%
ind = ones(size(dd,2),size(dd,3));
for i=1:period:nangles,
    dcf(:,i:i+2)=0;
    dd(:,:,i:i+2)=0;
    ind(:,i:i+2)=0;
end
 
%%
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
  clear tmp;
%%
mtx_reco = 200;
load(wf_name);

clear params;
params.dcf = dcf(:,1:end);
params.dd = dd_v(:,:,1:end);
params.phi = phi;%(:,1:end);
params.theta = theta;%(:,1:end);
params.nS = size(ks,2);
params.ks = ks*1.2*mtx_acq/mtx_reco; 
params.indices = squeeze(find(sum(abs(params.ks).^2,3)<0.25));


params.ks = params.ks(:,params.indices,:);
params.nks = length(params.ks);

params.mtx_reco = mtx_reco;

tform = affine3d(eye(4));
params = transformKspaceAndData(tform,params); 
temp = bsxfun(@times,transpose((params.dd)),(params.dcf(:)));
tic;initial_recon = params.FT'*temp;toc

ind = [-floor(mtx_reco/2):floor(mtx_reco/2)-1];
[x,y,z] = meshgrid(ind,ind,ind);
mask = x.^2+y.^2+z.^2 < 0.9*mtx_reco^2/4;
initial_sos = sqrt(sum(abs(initial_recon).^2,4)).*mask;
figure(1); imagesc(abs(initial_sos(:,:,mtx_reco/2)));title('initial sos');colormap(gray)

%% Try low resolution reconstruction
%load uniformangles
mtx_reco = 70;

ind = [-floor(mtx_reco/2):floor(mtx_reco/2)-1];
[x,y,z] = meshgrid(ind,ind,ind);
mask = x.^2+y.^2+z.^2 < 0.9*mtx_reco^2/4;


tform = affine3d;
plr = giveOperator(tform,phi,theta,ks,dcf,mtx_acq,mtx_reco,1.2,dd_v);

temp = bsxfun(@times,transpose(plr.dd),plr.dcf(:));
tic;reconlow = plr.FT'*temp;toc
soslow = sqrt(sum(abs(reconlow).^2,4));
figure(4);imagesc(abs(soslow(:,:,50)));title('lowres')

k = params.k;
temp = bsxfun(@times,transpose(plr.dd),plr.dcf(:));
T=1;
Tend = T+period*plr.nks-1;
W = zeros(1,period*plr.nks*79);
W(T:Tend)=1;
img1 = plr.FT'*bsxfun(@times,temp,W(:));
img1 = img1.*mask;
img1 = sqrt(sum(abs(img1).^2,4));     
angles = zeros(79,6);
phase = ones(size(k,1),1);

for n=2:78,
      Thigh = n*period*params.nks;
      Thighend = Thigh+period*params.nks-1;
      
      Tlow = n*period*plr.nks;
      Tlowend = Tlow+period*plr.nks-1;
      
      W = zeros(1,period*plr.nks*79);W(Tlow:Tlowend)=1;
      
      img2 = plr.FT'*bsxfun(@times,temp,W(:));
      img2 = img2.*mask;
      img2 = sqrt(sum(abs(img2).^2,4));      
      imagesc(abs(img2(:,:,floor(plr.mtx_reco/2))));drawnow
      [t3,tMatlab] = estRigid(abs(img2),abs(img1));
      
      deformParams = [rotm2eul(t3.T(1:3,1:3))'*180/pi,t3.T(4,1:3)*params.mtx_reco/plr.mtx_reco]
      [k(Thigh:Thighend,:),phase(Thigh:Thighend)] = rotateKspaceAndData(k(Thigh:Thighend,:),deformParams);
      angles(n,:) = deformParams;
end

params_corrected = updateKspaceTraj(params,k);

temp = bsxfun(@times,transpose(params_corrected.dd),conj(phase));
temp = bsxfun(@times,temp,params_corrected.dcf(:));
tic;reconCorrected = params_corrected.FT'*temp;toc
reconCorrected = sqrt(sum(abs(reconCorrected).^2,4));      
figure(6);imagesc(abs(reconCorrected(:,:,120)),[0,5e6]);title('Corrected');


%%
tic;reconCorrected = params.FT'*temp;toc
sz = size(reconCorrected(:,:,:,1));
%temp = bsxfun(@times,transpose(params.dd),conj(phase));
temp = transpose(params.dd);
tic;reconCorrected = params.FT'*temp;toc

A = @(x) reshape(params.FT'*(params.FT*reshape(x,sz)),prod(sz),1);
test = reconCorrected(:,:,:,1);
out = pcg(A,test(:),1e-4,100);
out = reshape(out,sz);
imagesc(abs(out(:,:,100)));

%%
for n=2:78,
      Thigh = n*period*params_corrected.nks;
      Thighend = Thigh+period*params_corrected.nks-1;
      
      W = zeros(1,period*params_corrected.nks*79);W(Thigh:Thighend)=1;
      
      img2 = params_corrected.FT'*bsxfun(@times,temp,W(:));
      imagesc(abs(img2(:,:,params_corrected.mtx_reco/2)));pause(0.1)
end

%%
% %%
% Vbasis = zeros(79,3);
% Vbasis(1:n1,1) = 1;
% Vbasis(n1+1:n2,2) = 1;
% Vbasis(n2+1:end,3) = 1;
% 
% Vbasisnorm = Vbasis*(Vbasis'*Vbasis)^(-1/2);
% recon = {};
% for i=1:size(Vbasis,2),
%     W = Vbasisnorm(:,i)*ones(1,period*size(plr.indices,2)); W = W';
%     temp = bsxfun(@times,plr.dd(:),plr.dcf);
%     temp = bsxfun(@times,temp,W(:));
%     recon{i} = plr.FT'*temp;
% end
% 
% img1 = zeros(plr.mtx_reco,plr.mtx_reco,plr.mtx_reco);
%     i=1;
%     for j=1:size(Vbasis,2),
%         img1 = img1 + Vbasis(i,j)*recon{j};
%     end
%         figure(i);imagesc(abs(img1(:,:,50)));title(num2str(i));pause(0.1);
% 
% for i=23:23:78,
%         test = zeros(plr.mtx_reco,plr.mtx_reco,plr.mtx_reco);
%     for j=1:size(Vbasis,2),
%         test = test + Vbasis(i,j)*recon{j};
%     end
%     figure(i);imagesc(abs(test(:,:,50)));title(num2str(i));pause(0.1);
%     
%     [t3,tMatlab] = estRigid(abs(test),abs(img1));
%     rotm2eul(t3.T(1:3,1:3))'*180/pi
%     %t3.T(4,1:3)
% end
% %%
%   k = params.k;
% 
%   k(T1+1:T2,:) = rotateKspace(k(T1+1:T2,:),[9.9878,9.9693,0.002]);
%   k(T2+1:end,:) = rotateKspace(k(T2+1:end,:),[-9.9934,-10.0639,-0.0271]);
%   params_corrected = updateKspaceTraj(params,k);
%   temp = bsxfun(@times,params_corrected.dd(:),params_corrected.dcf(:));
%   tic;reconCorrected = params_corrected.FT'*temp;toc
%   p2 = p2.*mask;
%   figure(6);imagesc(abs(p2(:,:,130)),[0,4e5]);title('Corrected');

