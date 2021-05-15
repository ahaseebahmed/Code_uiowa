
function [S1, b] = interpolateRadialtoCartesian(vKSpaceData,param)

%=================================================

% Input Parameters:

% vKSpaceData: raw data read from .dat file. Size : num_readouts X num_coils X num_lines

% n_lines, param.common_lines: Pattern of navigators is specified by these 2 parameters.
% If pattern is the following: 4 nav lines, 6 golden angle lines, 4 nav lines, 6 golden angle lines, ............
% Then n_lines = 10, param.common_lines = 4
% If acquisition is Golden Angle, then n_lines = param.lines_per_SG_block, param.common_lines = 0

% param.lines_per_SG_block: Number of lines per frame
% Suppose you have the 4,6,4,6,... pattern as before then for pre -interpolating data
% from reconstruction you should set param.lines_per_SG_block = 10

%=================================================

nChannels=size(vKSpaceData,2);
n=size(vKSpaceData,1);


%% calculate the number of time frames
ar =[1:param.lines_per_SG_block:size(vKSpaceData,3)];
param.nf = length(ar); % number of time frame

% bin the kspace data according to the above nrays and param.nf
knew = zeros(size(vKSpaceData,1),param.lines_per_SG_block,param.nf,nChannels);
for ch=1:nChannels
    for fr = 1:param.nf
        knew(:,:,fr,ch) = squeeze(vKSpaceData(:,ch,ar(fr):ar(fr)+param.lines_per_SG_block-1));
    end
end


[X, Y]=giveRadialTrajectory(param.n,param.lines_per_SG_block,param.common_lines,param.lines_to_skip,param.nf);

%% preinterpolation of the radial data onto Cartesian grid
S = cell(param.nf,nChannels);
b = cell(param.nf,nChannels);
nf = param.nf;

parfor ch = 1:nChannels
    fprintf('Preinterpolating Channel no: %d\n',ch)
    
    kdata_gr = squeeze(knew(:,:,:,ch));
    [kCART]=preinterpolateOneCoil(X(:,param.common_lines+1:end,:),Y(:,param.common_lines+1:end,:),kdata_gr(:,param.common_lines+1:end,:));
    
    kCART = fftshift(fftshift(kCART,1),2);
    kCART = reshape(kCART,n^2,nf);
    
    for i=1:nf
        S{i,ch} =find(kCART(:,i)~=0);
        b{i,ch} = squeeze(kCART(S{i,ch},i));
    end
    
end

S1 = S(:,1);

