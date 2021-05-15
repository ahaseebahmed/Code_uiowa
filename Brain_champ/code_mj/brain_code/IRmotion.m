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

%d='/Users/jcb/abdul_brain/31Jan20/P97792_0.75Res_IR.7';
%wf_name = '/Users/jcb/abdul_brain/31Jan20/radial3D_1H_fov192_mtx256_intlv206625_kdt4_gmax24_smax120_dur1_coca.mat';
d = '/Users/jcb/abdul_brain/scan17FebHuman/P88576.7'
wf_name = '/Users/jcb/abdul_brain/scan17FebHuman/radial3D_1H_fov200_mtx800_intlv206550_kdt6_gmax16_smax119_dur3p2_coca'


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

dcf(:,1:period:end)=0;
dcf(:,2:period:end)=0;
dcf(:,3:period:end)=0;

%% Initial reconstruction

nudgefactor = 1.1;
mtx_reco = 100;

osf = 2; wg = 3; sw = 8; 
k1= k*nudgefactor*mtx_acq/mtx_reco;
nangles = size(dd,3);

kmag = max(sqrt(sum(abs(k1).^2,3)),[],2);
indices = kmag<0.5;
k1 = k1(indices,:,:);
nsamplesNew = size(k1,1);

sz = size(k1);
ktemp = reshape(k1,prod(sz(1:2)),3);

FT = gpuNUFFT(transpose(ktemp),ones(1,prod(sz(1:2))),osf,wg,sw,[1 1 1]*mtx_reco,[],true);
dtemp = reshape(dd(:,indices,:),[nCh,nsamplesNew*nangles]);
dcftemp = reshape(dcf(indices,:),[1,nsamplesNew*nangles]);
tic;coilImages = FT'*bsxfun(@times,dtemp,dcftemp)';toc
sos = sqrt(sum(abs(coilImages).^2,4));
imagesc(imrotate(squeeze(abs(sos(floor(mtx_reco/2),:,:))),-90))

%%
disp('Coil Compression');

% use this step to exclude the dielectric pad
tmp = coilImages;


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
load(wf_name);

nWraps = 15;
nangles = size(dd,3);
nAcqsInWrap = floor(nangles/nWraps);
nSpokesinIR=375;
n_IRs = floor(nangles/nSpokesinIR);

clear k2
clear dcf1;
clear phiSorted;
clear thetaSorted;

%dd_v1 = zeros(nvCh,nsamples,nSpokesinIR,210);
startSpoke = 1;
for nIR=1:floor(nangles/nSpokesinIR) 
    dd_1(:,:,:,nIR) = reshape(dd_v(:,:,startSpoke:startSpoke+nSpokesinIR-1),[nvCh,nsamples,nSpokesinIR,1]);
    k2(:,:,nIR,:) = k(:,startSpoke:startSpoke+nSpokesinIR-1,:);
    dcf1(:,:,nIR) = dcf(:,startSpoke:startSpoke+nSpokesinIR-1);
    phiSorted(:,nIR) = phi(startSpoke:startSpoke+nSpokesinIR-1);
    thetaSorted(:,nIR) = theta(startSpoke:startSpoke+nSpokesinIR-1);
    
    startSpoke = startSpoke+nSpokesinIR;
    
    if(mod(startSpoke+nSpokesinIR,nAcqsInWrap)<=nSpokesinIR)
        startSpoke = startSpoke+2*nSpokesinIR-mod(startSpoke+nSpokesinIR,nAcqsInWrap);
    end
    if(startSpoke>nangles)
        break
    end
end

dcf1 = reshape(dcf1,nsamples,nSpokesinIR*nIR);

plr = giveOperator(affine3d,phiSorted(:),thetaSorted(:),ks,dcf1,mtx_acq,100,1.2,dd_1);
temp = bsxfun(@times,transpose(plr.dd),plr.dcf(:));
tic;reconlow = plr.FT'*temp;toc
soslow = sqrt(sum(abs(reconlow).^2,4));
figure(4);imagesc(abs(soslow(:,:,plr.mtx_reco/2)));title('lowres')

phr = giveOperator(affine3d,phiSorted(:),thetaSorted(:),ks,dcf1,mtx_acq,200,1.2,dd_1);
temp = bsxfun(@times,transpose(phr.dd),phr.dcf(:));
tic;reconhigh = phr.FT'*temp;toc
soshigh = sqrt(sum(abs(reconhigh).^2,4));
figure(4);imagesc(abs(soshigh(:,:,phr.mtx_reco/2)));title('highres')
%%

ind = [-floor(mtx_reco/2):floor(mtx_reco/2)-1];
[x,y,z] = meshgrid(ind,ind,ind);
mask = x.^2+y.^2+z.^2 < 0.9*mtx_reco^2/4;


temp = bsxfun(@times,transpose(plr.dd),plr.dcf(:));
T=1;
Tend = T+nSpokesinIR*plr.nks-1;
W = zeros(1,nSpokesinIR*plr.nks*nIR);
W(T:Tend)=1;
img1 = plr.FT'*bsxfun(@times,temp,W');
%img1 = img1.*mask;
img1 = sqrt(sum(abs(img1).^2,4));     
angles = zeros(nIR,6);
phase = ones(size(k,1),1);

for n=2:30:nIR,
%      Thigh = n*nSpokesinIR*params.nks;
 %     Thighend = Thigh+period*params.nks-1;
      
      Tlow = n*nSpokesinIR*plr.nks;
      Tlowend = Tlow+30*nSpokesinIR*plr.nks-1;
      
      W = zeros(1,nSpokesinIR*plr.nks*nIR);W(Tlow:Tlowend)=1;
      
      img2 = plr.FT'*bsxfun(@times,temp,W(:));
      %img2 = img2.*mask;
      img2 = sqrt(sum(abs(img2).^2,4));      
      imagesc(abs(img2(:,:,floor(plr.mtx_reco/2))));title(num2str(n));drawnow
     % [t3,tMatlab] = estRigid(abs(img2),abs(img1));
      
    %  deformParams = [rotm2eul(t3.T(1:3,1:3))'*180/pi,t3.T(4,1:3)*params.mtx_reco/plr.mtx_reco]
   %   [k(Thigh:Thighend,:),phase(Thigh:Thighend)] = rotateKspaceAndData(k(Thigh:Thighend,:),deformParams);
    %  angles(n,:) = deformParams;
end
%%

params_corrected = updateKspaceTraj(params,k);

temp = bsxfun(@times,transpose(params_corrected.dd),conj(phase));
temp = bsxfun(@times,temp,params_corrected.dcf(:));
tic;coilImages = params_corrected.FT'*temp;toc
reconCorrected = sqrt(sum(abs(coilImages).^2,4));      
figure(6);imagesc(abs(reconCorrected(:,:,120)));title('Corrected');