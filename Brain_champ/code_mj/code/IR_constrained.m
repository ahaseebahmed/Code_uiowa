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

%d='/Users/jcb/abdul_brain/31Jan20/P97792_0.75Res_IR.7';
%wf_name = '/Users/jcb/abdul_brain/31Jan20/radial3D_1H_fov192_mtx256_intlv206625_kdt4_gmax24_smax120_dur1_coca.mat';

[dd,k,dcf,mtx_acq,nangles,fov,phi,theta]  = readradial(d,[],wf_name);

nsamples = length(dcf)/nangles;
%c_ind=[2:7,10:15,18,19,22,23,26:27,30,31];
%dd=dd(c_ind,:);
nCh = size(dd,1);
dd = reshape(dd,nCh,nsamples,nangles);
k = reshape(k,nsamples,nangles,3);
dcf = reshape(dcf,nsamples,nangles);
ind = ones(size(dd,2),size(dd,3));


indices = find((phi==0) & (theta==0));
period = indices(5)-indices(4);

dcf = reshape(dcf,nsamples,nangles);
dcf(:,period:period:end)=0;
dcf(:,period-1:period:end)=0;
dcf(:,period-2:period:end)=0;
dcf(:,51:50:end)=0;
dcf(:,52:50:end)=0;
dcf(:,53:50:end)=0;


%% Initial reconstruction

nudgefactor = 1.1;
mtx_reco = 100;
mtx_reco = mtx_reco+mod(mtx_reco,2);

osf = 2; wg = 3; sw = 8; 
k1= k*nudgefactor*mtx_acq/mtx_reco;
nangles = size(dd,3);

kmag = max(sqrt(sum(abs(k1).^2,3)),[],2);
indices = kmag<0.5;
k1 = k1(indices,:,:);
nsamplesNew = size(k1,1);

sz = size(k1);
ktemp = reshape(k1,prod(sz(1:2)),3);

FT = gpuNUFFT(transpose(ktemp),ones(prod(sz(1:2)),1),osf,wg,sw,[1 1 1]*mtx_reco,[],true);
dtemp = reshape(dd(:,indices,:),[nCh,nsamplesNew*nangles]);
dcftemp = reshape(dcf(indices,:),[1,nsamplesNew*nangles]);
tic;coilImages = FT'*bsxfun(@times,dtemp,dcftemp)';toc
sos = sqrt(sum(abs(coilImages).^2,4));

%
% nCh = size(dd,1);
% dd = reshape(dd,nCh,nsamples,nangles);
% k = reshape(k,nsamples,nangles,3);
% dcf = reshape(dcf,nsamples,nangles);
% ind = ones(size(dd,2),size(dd,3));

%
disp('Coil Compression');

% use this step to exclude the dielectric pad
tmp = coilImages(:,:,30:end,:);


[n1,n2,n3,~] = size(tmp);
tmp = reshape(tmp,n1*n2*n3,nCh);
Rs = (real(tmp'*tmp));

[v,s] = eig(Rs);
s = diag(s);[s,i] = sort(s,'descend');
v = v(:,i);
s=s./sum(s);s = cumsum(s);
nvCh = min(5,nCh);%min(find(s>0.9));

dd_v = (v(:,1:nvCh)'*reshape((dd),nCh,nsamples*nangles));  
dd_v = reshape(dd_v,nvCh,nsamples,nangles);
  
%% Scanner based sorting
nudgefactor = 1.1;
mtx_reco = 200;

tosf = 1.25; % oversampling: 1.5 1.25
wg = 3; % kernel width: 5 7
sw = 8; % parallel sectors' width: 12 16
mtx_os = floor(mtx_reco*tosf);
mtx_os = mtx_os +sw - mod(mtx_os,sw);
osf = (mtx_os/mtx_reco);

k1= k*nudgefactor*mtx_acq/mtx_reco;

kmag = max(sqrt(sum(abs(k1).^2,3)),[],2);
indices = kmag<0.5;
k1 = k1(indices,:,:);
nsamplesNew = size(k1,1);

nWraps = 10;tosf = 1.25; % oversampling: 1.5 1.25
wg = 3; % kernel width: 5 7
sw = 8; % parallel sectors' width: 12 16
mtx_os = floor(mtx_reco*tosf);
mtx_os = mtx_os +sw - mod(mtx_os,sw);
osf = (mtx_os/mtx_reco);


nangles = size(dd,3);
nAcqsInWrap = floor(nangles/nWraps);
nSpokesinIR=350;
n_IRs = floor(nangles/nSpokesinIR);

clear dd_1
clear k2
clear dcf1;
%dd_v1 = zeros(nvCh,nsamples,nSpokesinIR,210);
startSpoke = 1;
for nIR=1:floor(nangles/nSpokesinIR) 
    dd_1(:,:,:,nIR) = reshape(dd_v(:,indices,startSpoke:startSpoke+nSpokesinIR-1),[nvCh,nsamplesNew,nSpokesinIR,1]);
    k2(:,:,nIR,:) = k1(:,startSpoke:startSpoke+nSpokesinIR-1,:);
    dcf1(:,:,nIR) = dcf(indices,startSpoke:startSpoke+nSpokesinIR-1);

    startSpoke = startSpoke+nSpokesinIR;
    
    if(mod(startSpoke+nSpokesinIR,nAcqsInWrap)<=nSpokesinIR)
        startSpoke = startSpoke+2*nSpokesinIR-mod(startSpoke+nSpokesinIR,nAcqsInWrap);
    end
    if(startSpoke>nangles)
        break
    end
end

dd_1 = dd_1(:,:,:,2:end);
nIRReduced = size(dd_1,4);
k2 = k2(:,:,2:end,:);
dcf1 = dcf1(:,:,2:end);

sz = size(k2);
ktemp = reshape(k2,prod(sz(1:3)),3);
FT = gpuNUFFT(transpose(ktemp),ones(prod(sz(1:3)),1),osf,wg,sw,[1 1 1]*mtx_reco,[],true);
dtemp = reshape(dd_1,[nvCh,prod(sz(1:3))]);
dcftemp = dcf1; 
dcftemp = reshape(dcftemp,1,prod(sz(1:3)));
tic;coilImages = FT'*bsxfun(@times,dtemp,dcftemp)';toc
sos = sqrt(sum(abs(coilImages).^2,4));
imagesc(imrotate(squeeze(abs(sos(floor(mtx_reco/2),:,:))),-90),[0,2e6])
    
[cmap,mask] = giveCSM(coilImages,5,12,0.05);%
cmap = bsxfun(@times,cmap,mask);

SOS = sum(coilImages.*conj(cmap),4);
figure(1);imagesc((abs(SOS(:,:,floor(mtx_reco/2)))),[0,0.25*max(abs(SOS(:)))]);


% Computation of Atb for each bin

%%

dcf2 = permute(dcf1,[1,3,2]);
sz = size(dcf2);
dcf2 = dcf2(:);
k2 = reshape(permute(k2,[1,3,2,4]),prod(sz),3);
dd_2 = reshape(permute(dd_1,[1,2,4,3]),nvCh,prod(sz));
FT = gpuNUFFT(transpose(k2),ones(prod(sz(1:3)),1),osf,wg,sw,[1 1 1]*mtx_reco,cmap,true);
test = FT'*(bsxfun(@times,dd_2,dcf2'))';
figure(1);imagesc((abs(test(:,:,floor(mtx_reco/2)))),[0,0.25*max(abs(test(:)))]);

T1 = [0.01,0.1,14,17,45,75,90],
T1 = log(2)./T1;

%%
Nbasis = 4;
t = 1:sz(3);t = t(:);
V = 1-2*exp(-t*T1);
[Vbasis,S] = svd(V,'econ'); 
Vbasis = Vbasis(:,1:Nbasis);
S = diag(S); 
S = S(1:Nbasis);
S = diag(1./S);

Atb = Atb_UV_T1(FT,dd_2,Vbasis,dcf2,mtx_reco);
M = @(x) AtA_UV_T1(FT,Atb,Vbasis,dcf2,mtx_reco,size(dd_2));
W = @(x) reshape(reshape(x,mtx_reco^3,size(Vbasis,2))*S,mtx_reco^3*size(Vbasis,2),1);
G = @(x) GradOp(x,size(Atb),[1,1,1,0]);
mu = 0.3;
Q = @(x) M(x) +  mu*G(x);

coeffs = pcg(Q,Atb(:),1e-8,10,[],[],Atb(:));
coeffs = reshape(coeffs,mtx_reco^3,size(Vbasis,2));
%%
for i=15:5:150%size(Vbasis,1)
    test = reshape(coeffs*Vbasis(i,:)',mtx_reco*[1,1,1]);
    imagesc(abs(test(:,:,mtx_reco/2)),[0,0.15*max(abs(test(:)))]); 
    title(num2str(i));
    pause(0.01);
end
%%

fname = ['/Shared/lss_jcb/jcb/scan21Feb/P76800_IR',num2str(mtx_reco),'_',num2str(mu)];
save(fname,'coeffs','Vbasis');