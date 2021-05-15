
function coilimages = giveCoilImages(kSpaceData,param,lines_to_use)

%=================================================

% Input Parameters:

% kSpaceData: raw data read from .dat file. Size : num_readouts X num_coils X num_lines

% n_lines, param.common_lines: Pattern of navigators is specified by these 2 parameters.
% If pattern is the following: 4 nav lines, 6 golden angle lines, 4 nav lines, 6 golden angle lines, ............
% Then n_lines = 10, param.common_lines = 4
% If acquisition is Golden Angle, then n_lines = param.lines_per_SG_block, param.common_lines = 0

% param.lines_per_SG_block: Number of lines per frame
% if you want to generate coil_sensitivity maps, you should set it to a 
% high value such as param.lines_per_SG_block = 1000 to avoid artefacts in sensitivity maps

%=================================================

nChannels=size(kSpaceData,2);
n=size(kSpaceData,1);
% calculate the number of time frames
ar =[1:param.lines_per_SG_block:lines_to_use];
nframes = length(ar); % number of time frame

% bin the kspace data according to the above nrays and nframes
knew = zeros(size(kSpaceData,1),param.lines_per_SG_block,nframes,nChannels);
for ch=1:nChannels
    for fr = 1:nframes
        knew(:,:,fr,ch) = squeeze(kSpaceData(:,ch,ar(fr):ar(fr)+param.lines_per_SG_block-1));
    end
end

%% preinterpolation of the radial data onto Cartesian grid

kspacecoils = zeros(n,n,ch);
maskcoils = zeros(n,n,ch);

[X, Y] = giveRadialTrajectory(n,param.lines_per_SG_block,param.common_lines,param.lines_to_skip,nframes);

X = X(:,param.common_lines+1:end,:);
Y = Y(:,param.common_lines+1:end,:);
knew = knew(:,param.common_lines+1:end,:,:);

parfor ch = 1:nChannels
    if(param.verbosity>2), fprintf('Reconstructing coil image %d\n',ch); end

    kdata_gr = squeeze(knew(:,:,:,ch));
    [kCART]=preinterpolateOneCoil(X,Y,kdata_gr);
    mask = abs(kCART)>0;
    kspacecoils(:,:,ch) = fftshift(sum(kCART,3));
    maskcoils(:,:,ch) = fftshift(sum(mask,3));
end

wt = abs(maskcoils(:,:,1));
mask = wt > 0;
kspacecoils = kspacecoils.*mask./(wt + not(mask));
coilimages = ifft2(kspacecoils);
