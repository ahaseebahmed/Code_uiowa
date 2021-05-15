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
d = './../31Jan20/P68608_woMotion.7';
%d = './../17Jan20/20200113_141044_P32768.7'; 
%d = './17Jan20/P52224.7'
wf_name = './../31Jan20/radial3D_1H_fov200_mtx400_intlv63200_kdt8_gmax33_smax118_dur5p2_fsca.mat';
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
mtx_reco = 250;
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
params.mtx_reco = mtx_reco;

tform = affine3d(eye(4));
params = transformKspaceAndData(tform,params); 
temp = bsxfun(@times,transpose(params.dd),params.dcf(:));
tic;initial_recon = params.FT'*temp;toc

ind = [-floor(mtx_reco/2):floor(mtx_reco/2)-1];
[x,y,z] = meshgrid(ind,ind,ind);
mask = x.^2+y.^2+z.^2 < 0.9*mtx_reco^2/4;
initial_sos = sqrt(sum(abs(initial_recon).^2,4)).*mask;
figure(1); imagesc(abs(initial_sos(:,:,125)),[0,6e5]);title('initial sos');colormap(gray)
%%

%% S

% load(wf_name);
% indices = find((phi==0) & (theta==0));
% period = indices(5)-indices(4);
% phi = reshape(phi,[period,length(theta)/period]);
% theta = reshape(theta,period,length(theta)/period);
% 
% %extracting the navigators
% navphi = phi(1:3,:);
% navtheta = theta(1:3,:);
% theta = theta(4:end,:);
% phi = phi(4:end,:);
% 
% %sorting the spokes
% 
% [thetasortedarray,phisortedarray] = groupAndSort(theta,phi);
% 
% %putting the navigators back
% thetasortedarray = [navtheta;thetasortedarray];
% phisortedarray = [navphi;phisortedarray];
% 
% 
% thetasortedarray = thetasortedarray(:);
% phisortedarray = phisortedarray(:);
% 
% save uniformangles thetasortedarray phisortedarray

%% Generate synthetic data
nChSim = 1;
mtx_reco=250;
load uniformangles

indices = find((phisortedarray==0) & (thetasortedarray==0));
period = indices(5)-indices(4);

n1=23;
T1 = period*n1;
n2=53;
T2 = period*n2;

tform = affine3d;
p = giveOperator(affine3d(eye(4)),phisortedarray,thetasortedarray,ks,dcf,mtx_acq,mtx_reco,1.2);

k = p.k;
n1=13;
T1 = period*n1*p.nks;
n2=23;
T2 = period*n2*p.nks;
n3=43;
T3 = period*n3*p.nks;
n4=63;
T4 = period*n4*p.nks;

phase = ones(1,size(k,1));

% [k(T1+1:T2,:)] = rotateKspace(k(T1+1:T2,:),[35,35,0]);
% [k(T2+1:T3,:)] = rotateKspace(k(T2+1:T3,:),[0,35,-35]);
% [k(T3+1:T4,:)] = rotateKspace(k(T3+1:T4,:),[-35,-35,0]);
% [k(T4+1:end,:)] = rotateKspace(k(T4+1:end,:),[0,-35,35]);

%[k(T1+1:T2,:),phase(T1+1:T2)] = rotateKspaceAndData(k(T1+1:T2,:),[10,10,0,0,0,0]);
%[k(T2+1:T3,:),phase(T2+1:T3)] = rotateKspaceAndData(k(T2+1:T3,:),[0,10,-10,0,0,0]);
%[k(T3+1:T4,:),phase(T3+1:T4)] = rotateKspaceAndData(k(T3+1:T4,:),[-10,10,0,0,0,0]);
%[k(T4+1:end,:),phase(T4+1:end)] = rotateKspaceAndData(k(T4+1:end,:),[0,-10,10,0,0,0]);

[k(T1+1:T2,:),phase(T1+1:T2)] = rotateKspaceAndData(k(T1+1:T2,:),[1,2,0,-2,1,0],true);
[k(T2+1:T3,:),phase(T2+1:T3)] = rotateKspaceAndData(k(T2+1:T3,:),[0,2,-1,0,-1,.5]);
[k(T3+1:T4,:),phase(T3+1:T4)] = rotateKspaceAndData(k(T3+1:T4,:),[-1,1,0,1,2,0],true);
[k(T4+1:end,:),phase(T4+1:end)] = rotateKspaceAndData(k(T4+1:end,:),[0,-2,1,1,2,0]);


% [k(T1+1:T2,:),phase(T1+1:T2)] = rotateKspaceAndData(k(T1+1:T2,:),[3,2,0,-4,3,0]);
% [k(T2+1:T3,:),phase(T2+1:T3)] = rotateKspaceAndData(k(T2+1:T3,:),[0,2,-1,0,-3,2]);
% [k(T3+1:T4,:),phase(T3+1:T4)] = rotateKspaceAndData(k(T3+1:T4,:),[-2,1,0,2,3,0]);
% [k(T4+1:end,:),phase(T4+1:end)] = rotateKspaceAndData(k(T4+1:end,:),[0,-2,1,5,6,0]);


%%[k,d] = rotateKspaceAndData(k,d(T1+1:T2,:),params);

%synthetic_data = transpose(p.FT*phantom3d(p.mtx_reco));
synthetic_data = transpose(p.FT*initial_sos);
%synthetic_data = transpose(p.FT*initial_recon);


temp = bsxfun(@times,synthetic_data,reshape(p.dcf,[1,p.nks*nangles]));
tic;recon1 = p.FT'*transpose(temp);toc
sos1 = sqrt(sum(abs(recon1).^2,4));
figure(2);imagesc(abs(sos1(:,:,p.mtx_reco/2)),[0,4e5]);title('Uncorrupted');colormap(gray)

p = updateKspaceTraj(p,k);

synthetic_data = bsxfun(@times,synthetic_data,phase);

raw = zeros(nChSim,p.nS,nangles);
raw(:,p.indices,:) = reshape(synthetic_data,[nChSim,p.nks,nangles]);

tform = affine3d;
%tform.T(1:3,1:3) = eul2rotm(pi/180*[10;10;0]);
phr = giveOperator(tform,phisortedarray,thetasortedarray,ks,dcf,mtx_acq,mtx_reco,1.2,raw);
temp = bsxfun(@times,transpose(phr.dd),phr.dcf(:));
tic;recon2 = phr.FT'*temp;toc
sos2 = sqrt(sum(abs(recon2).^2,4));
figure(3);imagesc(abs(sos2(:,:,p.mtx_reco/2)),[0,4e5]);title('Corrupted');colormap(gray)

%% Try low resolution reconstruction
load uniformangles
mtx_reco = 100;

ind = [-floor(mtx_reco/2):floor(mtx_reco/2)-1];
[x,y,z] = meshgrid(ind,ind,ind);
mask = x.^2+y.^2+z.^2 < 0.9*mtx_reco^2/4;


tform = affine3d;
plr = giveOperator(tform,phisortedarray,thetasortedarray,ks,dcf,mtx_acq,mtx_reco,1.2,raw);

temp = bsxfun(@times,transpose(plr.dd),plr.dcf(:));
tic;reconlow = plr.FT'*temp;toc
soslow = sqrt(sum(abs(reconlow).^2,4));
figure(4);imagesc(abs(soslow(:,:,50)));title('lowres')

k = phr.k;
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
      Thigh = n*period*phr.nks;
      Thighend = Thigh+period*phr.nks-1;
      
      Tlow = n*period*plr.nks;
      Tlowend = Tlow+period*plr.nks-1;
      
      W = zeros(1,period*plr.nks*79);W(Tlow:Tlowend)=1;
      
      img2 = plr.FT'*bsxfun(@times,temp,W(:));
      img2 = img2.*mask;
      img2 = sqrt(sum(abs(img2).^2,4));      
      imagesc(abs(img2(:,:,50)));drawnow
      [t3,tMatlab] = estRigid(abs(img2),abs(img1));
      
      deformParams = [rotm2eul(t3.T(1:3,1:3))'*180/pi,t3.T(4,1:3)*phr.mtx_reco/plr.mtx_reco]
      [k(Thigh:Thighend,:),phase(Thigh:Thighend)] = rotateKspaceAndData(k(Thigh:Thighend,:),deformParams);
      angles(n,:) = deformParams;
end

phr_corrected = updateKspaceTraj(phr,k);

temp = bsxfun(@times,transpose(phr_corrected.dd),conj(phase));
temp = bsxfun(@times,temp,phr_corrected.dcf(:));
tic;reconCorrected = phr_corrected.FT'*temp;toc
reconCorrected = sqrt(sum(abs(reconCorrected).^2,4));      
figure(6);imagesc(abs(reconCorrected(:,:,125)),[0,4e5]);title('Corrected');
%%

% k = phr.k;
% anglesnew = zeros(period*phr.nks*79,3);
% anglesnew(T1+1:T2,:) = bsxfun(@plus,anglesnew(T1+1:T2,:),[10,10,0]);
% anglesnew(T2+1:end,:) = bsxfun(@plus,anglesnew(T1+1:T2,:),[-10,-10,0]);
% 
% k(T1+1:T2,:) = rotateKspace(k(T1+1:T2,:),[10,10,0]);
% k(T2+1:end,:) = rotateKspace(k(T2+1:end,:),[-10,-10,0]);
% phr_actual = updateKspaceTraj(phr,k);
% temp = bsxfun(@times,transpose(phr_actual.dd),phr_actual.dcf(:));
% tic;reconActual = phr_actual.FT'*temp;toc
% reconActual = sqrt(sum(abs(reconActual).^2,4));      
% figure(7);imagesc(abs(reconActual(:,:,125)));title('Corrected');

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
%   k = phr.k;
% 
%   k(T1+1:T2,:) = rotateKspace(k(T1+1:T2,:),[9.9878,9.9693,0.002]);
%   k(T2+1:end,:) = rotateKspace(k(T2+1:end,:),[-9.9934,-10.0639,-0.0271]);
%   phr_corrected = updateKspaceTraj(phr,k);
%   temp = bsxfun(@times,phr_corrected.dd(:),phr_corrected.dcf(:));
%   tic;reconCorrected = phr_corrected.FT'*temp;toc
%   p2 = p2.*mask;
%   figure(6);imagesc(abs(p2(:,:,130)),[0,4e5]);title('Corrected');

