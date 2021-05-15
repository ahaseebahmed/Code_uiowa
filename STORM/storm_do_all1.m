 %function [U1,D,csm,vCoilImages,param,Ul2] = storm_do_all(param,kSpaceData)
%function [b, bCom_smoothed,vCoilImages,csm,S,L] = storm_do_all1(param,kSpaceData)
%function [L] = storm_do_all1(param,kSpaceData)
function [vKSpaceData] = storm_do_all1(param,kSpaceData)



% Default Parameters

if not(isfield(param, 'verbosity')),             param.verbosity = 2;      end
if not(isfield(param, 'CoilSolveCGIterations')), param.CoilSolveCGIterations = 50; end
if not(isfield(param, 'lambdaCoeffs')), param.lambdaCoeffs = 0.2; end
if not(isfield(param, 'LaplacianRegParam')), param.LaplacianRegParam = 1e-1; end
if not(isfield(param, 'common_lines')), param.common_lines = 4; end
if not(isfield(param, 'lines_per_SG_block')), param.lines_per_SG_block = 10; end
if not(isfield(param, 'blocks_to_skip')), param.blocks_to_skip = 100; end
if not(isfield(param, 'CoeffsCGIterations')), param.CoeffsCGIterations = 100; end
if not(isfield(param, 'T')), param.T = 0.05; end
if not(isfield(param, 'useGPU')), param.useGPU = 1; end

param.lines_to_skip = param.lines_per_SG_block*param.blocks_to_skip;

% skipping initial lines
%------------------------

kSpaceData = kSpaceData(:,:,param.lines_to_skip+1:end);
param.n = size(kSpaceData,1); % Size of each image is nxn
param.nchannels = size(kSpaceData,2); % Number of coils
param.NcoilSelect=param.nchannels;
param.nlines = size(kSpaceData,3); 
param.nf = min(param.nf, param.nlines/param.lines_per_SG_block);
kSpaceData = kSpaceData(:,:,1:param.nf*param.lines_per_SG_block);
%param.nf = param.nlines/param.lines_per_SG_block;
kSpaceData=kSpaceData./max(abs(kSpaceData(:)));
%% Estimation of coil images and preselection
coilimages1 = giveCoilImagesStartEnd(kSpaceData(),param,1,floor(param.nf/2));
coilimages2 = giveCoilImagesStartEnd(kSpaceData(),param,ceil(param.nf/2),param.nf);

%% coil combination
%------------------
if(param.verbosity), fprintf('Coil compressing the data\n'); end

[vKSpaceData,vCoilImages] = combine_coils_covar(kSpaceData,coilimages1,coilimages2,param,0.8);
nvchannels = size(vKSpaceData,2);

if(param.verbosity),fprintf('Computing ESPiRIT maps\n');end


% csm = giveEspiritMaps(vCoilImages);
% csm = fftshift(fftshift(csm,1),2);
% gwin = giveMask(vCoilImages,param.szSmooth);
% param.gwin = gwin;
% gwin = repmat(1+10*gwin,[1,1,param.Nbasis]);
%% Form matrix bCom to estimate Manifold Laplacian
if(param.verbosity),fprintf('Forming bCom\n');end
if param.common_lines >0
    kSpaceTemp = reshape(vKSpaceData,[param.n,nvchannels,param.lines_per_SG_block,param.nf]);
    bCom = kSpaceTemp(:,:,1:param.common_lines,:);
    clear kSpaceTemp
end

% if(param.verbosity),fprintf('Estimating the Laplacian\n');end
% 
% 
% bCom_img=fftshift(ifft(fftshift(bCom,1)),1);
% bCom_img = reshape(bCom_img,param.n*nvchannels*param.common_lines,param.nf);
% bCom_img=bCom_img./max(bCom_img(:));
% %
% q = fft(bCom_img,[],2);
% q(:,param.nf/2-param.nf/4-115:param.nf/2+param.nf/4+115)=0;
% bCom_smoothed = ifft(q,[],2);
% 
% [~, bCom_est, L] = estimateLapKernelLR(bCom_smoothed, 0.25, 0.1,0);
% [~,SBasis,D] = svd(L);
% D= D(:,end-param.Nbasis:end);
% SBasis = SBasis(end-param.Nbasis:end,end-param.Nbasis:end);
% 
% %% Pre-interpolate data
% 
% [S, b] = interpolateRadialtoCartesian(vKSpaceData,param);


%% Solving for the coefficients


%     if(param.verbosity),fprintf('Using L2 recovery\n');end
% 
%     c1 = AhbVh(b,S,D',conj(csm),param.n,size(D,2),param.nf,nvchannels,param.useGPU);
%     LamMtx = SBasis*param.lambdaCoeffs;
%     gradU = @(z)(AhAUVVh_reg(z,D',S,csm,param.n,size(D,2),param.nf,nvchannels,LamMtx,[],[],param.useGPU,gwin));
%     tic;[U,~,~,~,err] = pcg(gradU,c1,10^-10,param.CoeffsCGIterations);toc;
%     U1 = gather(reshape(U,[param.n^2,size(D,2)]));    
%     Ul2=[];
%    if(param.verbosity > 2), fprintf('Iter done in %d secs\n',toc);end
% 
%    if(param.SolverType=='L1')
%         if(param.verbosity),fprintf('Using L1 recovery\n');end
%                
%         for i=1:param.outerIterations
%               [reg_term] = rhs_reg(U,param.n,param.Nbasis+1,LamMtx, param.eps,gwin);
%               LamMtx = SBasis*5*param.lambdaCoeffs; 
%               gradU = @(z)(AhAUVVh_reg(z,D',S,csm,param.n,size(D,2),param.nf,nvchannels,LamMtx,[],[],param.useGPU,gwin));
%               tic;[U,~,~,~,err] = pcg(gradU,c1 + reg_term,10^-10,25,[],[],U);toc;
%               if(param.verbosity > 1), fprintf('Iter %d done in %f secs\n',i+1,toc);end
%         end
%     U1 = gather(reshape(U,[param.n^2,size(D,2)]));    
% 
% end
