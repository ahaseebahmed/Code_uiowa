function [x, V, kdata,dcf,ktraj_scaled] = spiral_storm_navigator_3d_updated( varargin)
%function img = spiral_storm_navigator_3d( varagin)
%function [img, Mv, Svalues, V, params, flags] = spiral_storm_navigator_3d( kdata, knav, ktraj, dcf, N, Nt, params, flags)
%
% prompt for mat file containing below arrays if nargin = 0
%
% Arguments:
%   kdata   [np nv ns nc]   complex
%   knav    [np nvn ns nc]  complex
%   ktraj   [3 np nv ns]
%   dcf     [np nv ns]
%
% Optional Arguments:
%   N               reconstucted image size
%   Nt              time frames
%   params          parameter structure for reconstruction
%   flags           flag structure for debug mode, etc.
%
% Outputs:
%   img     [N N N Nt]      complex
%   Mv      movie of abs( img(:, :, N/2, :))
%   Svalues singular values of the similarity matrix
%   V       singular vectors used
%   params  parameters used
%   flags   flags used
%
% Ahmed, Abdul Haseeb <abdul-ahmed@uiowa.edu>
% 3d version of spiralStorm_navigator, CAC 190312

%% Environment setup, parameters, logging and debugging
%clear all;      % warn about this first, *** CAC 190220

ss_nav_version = '190314.01';  % version
runtime_str = datestr( now);   %runtime_str = datestr( now, 'yymmdd_HHMMSS'); 
startPath = pwd;
%cd('spiral_storm_navigator_3d');


%img = []; Mv = []; Svalues = []; V = [];
params = [];    % need to put below parameters in structure, also handle argument and output, *** CAC 190313
flags.debug = 3;    %0 off, 1 timing, 2 plots, internal, 3 log/save, environment 4 more...
flags.warning = 1;  %0 off, 1 on

% Reconstruction parameters (defaults)
spiralsToDelete = 0;    % git rid of this for 3d, use framesToDelete, *** CAC 190314
framesToDelete = 0;
nGroup = 1;             % grop succesive data frames together, also temp fix for alternating navigators *** CAC 190308
%ninterleavesPerFrame = 192*nGroup;  % this may need fixing for multiCoil sections
N = 128;                 % reconstruction matrix size, also crops fid to N/2, CAC 190405
nChannelsToChoose = 1;  % starting number of virtual coils
numFramesToKeep = 999;  %numFramesToKeep = 64;
useGPU = 'true';        % 'false' not working yet
%SHRINK_FACTOR = .5*128/32;                  % ngfnRecon data is +-1, gpuNUFFT wants +-0.5 for k, truncating fid requires scaling
nBasis = min( [8 numFramesToKeep/2]);       % must be numFramesToKeep or smaller? Ignored if auto_nbasis = 1
navD = 1;               %navD = 8;              % decimation factor for navigator views, use every 1 of navD navigator views in segment.
auto_nbasis = 1;        %auto_nbasis = 0;       % automatically calculate the nBasis using threshold, if true
Sthresh_rel = 1.000e-4;   % relative threshold to keep for singular basis (works if snr high enough?), depends on snr
lambdaSmoothness = 0.00025;                  %lambdaSmoothness = 0.025;	% tuning parameter (time domain)
cRatioI = 1:nChannelsToChoose;
sigma = [0.45];         %sigma = [4.5];     % tuning parameter (~distance)
lam = [0.1];            %lam = [0.1];       % tuning parameter

% new parameters
nIterations = 200;      %nIterations = 60;      % iterations for final reconstuction
nIterations_Lap = 200;  %nIterations_Lap = 70;  % number of iterations for Lapace kernel calculation
nIterations_csm = 50;   %nIterations_csm = 70;  % iterations for coil sensitivity map
eigThresh_1 = 0.02;     %eigThresh_1 = 0.008:   % threshold for picking singular vercors of the calibration matrix (relative to largest singlular value.)
eigThresh_2 = 0.95;     %eigThresh_2 = 0.95;    % threshold of eigen vector decomposition in image space.

% handle arguments
switch nargin
    case {0}    % Interactive
        % below
    case {1, 2, 3}
        error( 'too few arguments');
    case {4}    % kdata, knav, ktraj, dcf
        kdata = varargin{1};
        knav = varargin{2};
        ktraj = varargin{3};
        dcf = varargin{4};
    case {5}    % N
        kdata = varargin{1};
        knav = varargin{2};
        ktraj = varargin{3};
        dcf = varargin{4};
        N = varargin{5};
    case {6}    % Nt
        kdata = varargin{1};
        knav = varargin{2};
        ktraj = varargin{3};
        dcf = varargin{4};
        N = varargin{5};
        numFramesToKeep = varargin{6};
    case {7}    % update params structure
        kdata = varargin{1};
        knav = varargin{2};
        ktraj = varargin{3};
        dcf = varargin{4};
        N = varargin{5};
        numFramesToKeep = varargin{6};
        params = varargin{7};
    case {6}    % update flags structure
        kdata = varargin{1};
        knav = varargin{2};
        ktraj = varargin{3};
        dcf = varargin{4};
        N = varargin{5};
        numFramesToKeep = varargin{6};
        params = varargin{7};
        flags = varargin{8};
    case {7}
        error( 'too many arguments');
end

if flags.debug >= 1; tstart = tic; end  

if flags.warning >= 1; warning('ON'); else warning('OFF'); end

if flags.debug >= 3; diary_file = strcat( 'spiral_storm_navigator_3d_', runtime_str, '.log'); diary( diary_file); end

if flags.debug >= 1; fprintf( '===== spiral_storm_navigator_3d version = %s =====\n', ss_nav_version);  end
if flags.debug >= 2; fprintf( 'runtime: %s\n', runtime_str); end
if ( 3 > flags.debug >= 2 ); fprintf('MATLAB Version: %s\n', version); end % print matlab version
if ( flags.debug >= 3 ); ver; end % print detailed matlab and system info
[git_status, result] = system( 'git status'); % check for git directory
if ( flags.debug >= 2 ) & ~git_status; fprintf( 'Git Information: '); system('git describe --all --dirty --long'); end
if ( flags.debug >= 3 ) & ~git_status; system('git branch -v --no-abbrev; git remote -v;'); end
if ( flags.debug >= 4 ) & ~git_status; system('git status -vv;'); end
if flags.debug >= 4; path,  end

if nargin == 0 % loading and reformatting data...
    if flags.debug >= 1; tload = tic; end
    if flags.debug >= 1; fprintf( 'loading and reformatting data...');  end
    %save and setup paths
    %mlp = path;
    %restoredefaultpath;
    addpath( './csm');
    addpath( './Utils');
    %addpath( './nufft_toolbox_cpu');
    %addpath( genpath( './gpuNUFFT'));
    %addpath( genpath( './CUDA'));
    % ln -s ../@ngfnRecon @ngfnRecon needed to load forSSN_obj.mat
    [fileName, pathName, filterIndex] = uigetfile( ...
        {'forssn_3d*.mat','MAT-files (forssn_3d*.mat)'}, ...
        'Pick a file containing kdata...');
    if filterIndex == 0
        warning('Cancelled by user');
        return;
    else      
        if flags.debug >= 2; fprintf( '%s, ', fileName); end
        load( fileName);
        %kdata = forSSN_obj.fid;
        %knav = forSSN_obj.fid_s;
        %ktraj = permute( forSSN_obj.kspaceRad, [4 1 2 3]);
        %dcf = forSSN_obj.dcf;
        
        if flags.debug >= 1; toc( tload), end
    end
end

%% =========================================
% -------------Preprocessing Data-------------%
%===========================================
if flags.debug >= 1; tproc = tic; end
if flags.debug >= 1; fprintf( 'preprocessing data...');  end

% Arguments:
%   kdata   [np nv ns nc]   complex
%   knav    [np nvn ns nc]  complex
%   ktraj  [3 np nv ns]
%   dcf     [np nv ns]

[nFreq, ninterleavesPerFrame, numFrames, nCh] = size( kdata);
%numberSpirals = ninterleavesPerFrame*numFrames;
nFreqEncoding = min( [nFreq N/2]);
numFramesToKeep = min( [numFramesToKeep numFrames/nGroup]);
nChannelsToChoose = min( [nChannelsToChoose nCh]);

kdata=reshape(kdata,[nFreq,ninterleavesPerFrame,numFrames]);
%kdata=reshape(kdata,[nFreq,ninterleavesPerFrame*4,numFramesToKeep]);

ktraj=reshape(ktraj,[3,nFreq,ninterleavesPerFrame,numFrames]);
%ktraj=reshape(ktraj,[3,nFreq,ninterleavesPerFrame*4,numFramesToKeep]);

dcf=reshape(dcf,[nFreq,ninterleavesPerFrame,numFrames]);
%dcf=reshape(dcf,[nFreq,ninterleavesPerFrame*4,numFramesToKeep]);


kdata = kdata(1:nFreqEncoding, :, (framesToDelete + 1):(numFramesToKeep + framesToDelete), cRatioI(1:nChannelsToChoose));
ktraj = ktraj(:, 1:nFreqEncoding, :, (framesToDelete + 1):(numFramesToKeep + framesToDelete)); %size( ktraj) % breaks 2d
dcf = dcf(1:nFreqEncoding, :, (framesToDelete + 1):(numFramesToKeep + framesToDelete)); %size( dcf) % breaks 2d

if flags.debug >= 1; toc( tproc), end

%% ==============================================================
% Scaling trajectory
% ==============================================================

%max( abs( ktraj), [], 'all')

SHRINK_FACTOR = 1;%0.5 / max( abs( ktraj), [], 'all');
ktraj_scaled =  SHRINK_FACTOR*ktraj*N; %max( abs( ktraj_scaled), [], 'all')

%return

%% ==============================================================
% Compute the coil sensitivity map
% ==============================================================
if nChannelsToChoose >= 2
    if flags.debug >= 1; tcsm = tic; end
    if flags.debug >= 1; fprintf( 'computing coil sensitivity map for %d channels...', nChannelsToChoose);  end
    
    ktraj_scaled = reshape( ktraj_scaled, [3, nFreqEncoding, ninterleavesPerFrame, numFramesToKeep]); % breaks 2d
    kdata = reshape( kdata, [nFreqEncoding, ninterleavesPerFrame, numFramesToKeep, nChannelsToChoose]);
    
    [coilImages] = coil_sens_map_NUFFT( kdata, ktraj_scaled, N, useGPU, nIterations_csm);
    
    if flags.debug >= 1; toc( tcsm), end
else
    if flags.debug >= 1; fprintf( 'single channel data\n');  end
end

%% ===============================================================
% Compute coil compresession
% ================================================================
if nChannelsToChoose >= 2
    if flags.debug >= 1; tccc = tic; end
    if flags.debug >= 1; fprintf( 'computing coil compression...');  end
    
    kdata = reshape( kdata, [nFreqEncoding*ninterleavesPerFrame*numFramesToKeep, nChannelsToChoose]);
    
    [vkdata, vcoilImages] = combine_coils( kdata, coilImages, 0.85); % 0.85 parameter in variable, *** CAC 190220
    nChannelsToChoose = size( vcoilImages, 3);
    kdata = reshape( vkdata, [nFreqEncoding, ninterleavesPerFrame, numFramesToKeep, nChannelsToChoose]);
    csm = giveEspiritMaps( reshape( vcoilImages, [size( vcoilImages, 1), size( vcoilImages, 2), nChannelsToChoose]), eigThresh_1, eigThresh_2);
    coilImages = vcoilImages;
    
    ktraj_scaled = reshape( ktraj_scaled, [3, nFreqEncoding, ninterleavesPerFrame, numFramesToKeep]); %size( ktraj_scaled)
    kdata = reshape( kdata, [nFreqEncoding, ninterleavesPerFrame, numFramesToKeep, nChannelsToChoose]);
    
    if flags.debug >= 1; toc( tccc), end
else
    % anything else needed?
    csm = ones( N, N, N, nCh); % breaks 2d
end

%% ==============================================================
% % Compute the weight matrix
% % =============================================================
if flags.debug >= 1; tcwm = tic; end
if flags.debug >= 1; fprintf( 'computing weight matrix...');  end

if nChannelsToChoose >= 2
    no_ch = size( csm, 3); % breaks 2d
else
    no_ch = 1;
end

[~, nViewsNav, ~, ~] = size( knav); nViewsNavKeep = nViewsNav/navD; %nViewsNavKeep should be an integer
knav = knav(1:nFreqEncoding, 1:navD:end, (framesToDelete + 1):end, cRatioI(1:nChannelsToChoose)); %size( knav)
knav = permute( knav, [1 2 4 3]); %size( knav) % which view to use for navigator, possible to use all in sphere? *** CAC 1090304
knav = knav(:, :, :, 1:nGroup:end); %size( knav)
knav = knav(:, :, :, 1:numFramesToKeep); %size( knav)

%uncomment for tuning matrix of recons
%
% ss2 = size( sigma, 2)
% sl2 = size( lam, 2)
% for ii = 1:size( sigma, 2)
%     for jj = 1:size( lam, 2)
%[~, ~, L] = estimateLapKernelLR( reshape( knav, [nFreqEncoding*no_ch*nViewsNavKeep, numFramesToKeep]), sigma(ii), lam(jj), nIterations_Lap);

[~, ~, L] = estimateLapKernelLR( reshape( knav, [nFreqEncoding*no_ch*nViewsNavKeep, numFramesToKeep]), sigma(1), lam(1), nIterations_Lap); %size( L)
[~, Sbasis, V] = svd( L); %size( Sbasis)

Svalues = diag( Sbasis);
ssize = size( Sbasis, 1);
if auto_nbasis
    Smax = max( Svalues); Smin = min( Svalues);
    Sthresh = Smax*Sthresh_rel; nBas = 0;
    for idx = size( Svalues):-1:1
        if Svalues(idx) < Sthresh
            nBas = nBas + 1;
        end
    end
    nBas = min( [numFramesToKeep/2 nBas]);
    nBasis = max( [4 nBas]);
end

V = V(:, (end - nBasis + 1):end); %size( V)
Sbasis = Sbasis((end - nBasis + 1):end, (end - nBasis + 1):end); %size( Sbasis)

if ( flags.debug >= 2 )
    for nidx = 1:nBasis; txt = sprintf('singular vector: %d', numFramesToKeep + nidx - nBasis); figure( 'Name', txt); plot( V(:, nidx), '-x'); title( txt); end
    figure( 'Name', 'Relative Log10 Svalues'); title( 'Relative Log10 Svalues'); hold on;
    plot( 1:(ssize-nBasis), log10( Svalues(1:(end-nBasis))/Svalues(1)), '-+r');
    plot( (ssize-nBasis+1):ssize, log10( Svalues((end-nBasis+1):end)/Svalues(1)), '-+g'); hold off;
end

if flags.debug >= 1; toc( tcwm), end

if flags.debug >= 2;
    reply = input('Do you want to continue? Y/N [Y]:','s');
    if isempty(reply)
        reply = 'Y';
    end
    if reply == 'y' || reply == 'Y'
        % continue
    else
        cd(startPath); return     % just look at Svalues and Sbasis
    end
end

%% ==============================================================
% % Final Reconstruction
% % =============================================================
if flags.debug >= 1; tfr = tic; end
if flags.debug >= 1; fprintf( 'performing SToRM reconstruction of 3d size %d, frames %d, with nBasis %d ...', N, numFramesToKeep, nBasis);  end

ktraj_scaled = reshape( ktraj_scaled, [3, nFreqEncoding*ninterleavesPerFrame, numFramesToKeep]); %skt = size( ktraj_scaled) % breaks 2d
dcf = reshape( dcf, [nFreqEncoding*ninterleavesPerFrame, numFramesToKeep]); %s_w = size( w) % breaks 2d
kdata = reshape( kdata, [nFreqEncoding*ninterleavesPerFrame, numFramesToKeep, nChannelsToChoose]); %skd = size( kdata)

%scsm = size( csm), sV = size( V)

x = solveUV( ktraj_scaled, dcf, kdata, csm, V, N, nIterations, lambdaSmoothness*Sbasis, useGPU); % nIterations now variable at top, CAC 190220

img = reshape( reshape( x, [N*N*N, nBasis]) * V', [N, N, N, numFramesToKeep]); % breaks 2d

% if flags.debug >= 1; toc( tfr), end
% 
% % ==============================================================
% % Save and Display results
% % =============================================================
% 
% for i=1:530;imagesc(fliplr(flipud(abs(y(:,:,i))))); pause(0.1); colormap gray;end
% for i=1:250;imagesc(((abs(y(:,:,i))))); pause(0.1); colormap gray; end
% clear kdata csm V ;
% 
% bug, velow commented out, what is var 'sl', CAC 190219
% save(strcat('res_iter_',num2str(lambdaSmoothness),'_',num2str(sigma(ii)),'_',num2str(sl),'.mat'),'y','-v7.3');
% 
% cd './../../../SpiralcodeUVv2/SpiralcodeUV_new';
%     end
% end
% 
% movie display, CAC 190219
% if flags.debug >= 1; tmv = tic; end
% if flags.debug >= 2;
%     fprintf( 'making movie...');
%     
%     simg = size( img);
%     figure( 'Name', 'Movie of center slice');
%     hold on;
%     for idx_t = 1:simg(4) % breaks 2d
%         colormap gray;
%         imagesc( abs( img(:, :, end/2, idx_t)));
%         Mv(idx_t) = getframe;
%     end
%     movie( Mv, 2);
%     hold off;
%     
%     save of .mov file or other would go here, *** CAC 190221
%     
%     toc( tmv)
% end
% 
% save everything, CAC 190220
% if flags.debug >= 3; tsave = tic; end
% if flags.debug >= 3;
%     save_file = strcat( 'spiral_storm_navigator_3d_', runtime_str, '.mat');
%     fprintf( 'saving results file: %s ...', save_file);
%     save( save_file, '-mat', '-v7.3');
% end
% if flags.debug >=  3; toc( tsave), end
% 
% % restore environment
% if exist('RESTOREDEFAULTPATH_EXECUTED');
%     if RESTOREDEFAULTPATH_EXECUTED;
%         path( mlp);
%     end
% end
% cd(startPath);
% 
% % total elapsed time
% if flags.debug >= 1; fprintf( 'SToRM Total ');toc( tstart), end
% if flags.debug >= 3; diary off; fprintf( 'saving log file: %s\n', diary_file); end

end

% v=VideoWriter('co1.avi')
% v.FrameRate=10
% open(v)
% for i=1:32
% writeVideo(v,xx(:,:,i));
% end
% close(v)
%addpath('/Users/ahhmed/registraion/antsbin/ANTS-build')