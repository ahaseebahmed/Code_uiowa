% Script for reading the .dat file with 3D radial rawdata specified in 
% "rawdata_path" and computing the corresponding trajectory

clear all;
close all;
clc;

%% Setup paths
restoredefaultpath;

% Add path to mapVBVD
addpath(genpath('mapVBVD'))


%% Specify rawdata
rawdata_path = 'meas_MID00870_FID98515_BEAT_SelfNav_1032_HighRes.dat';


%% Read Siemens twix object
twix_obj = mapVBVD(rawdata_path);

% Select imaging data
if iscell(twix_obj)
    twix_obj = twix_obj{end};
end

twix_obj.image.flagIgnoreSeg = true; % Ignore segments: necessary for 3D radial

% Read k-space data
kspace = twix_obj.image{''};

size(kspace) %[Readout Coils Lines]

%%
%****** ACQUISITION PARAMETERS ******%
% twix_obj.image.NCol                         Number of samples per line (readout)
% twix_obj.image.NSeg                         Number of heartbeats (shots)
% twix_obj.image.NLin/twix_obj.image.NSeg     Number of segments per heartbeat (segments)
% twix_obj.image.NLin                         Total number of lines (readouts)
% twix_obj.image.NCha                         Number of coils

% reshape kspace data to [Readout Segments Shots Coils]
kspace = permute(kspace, [1 3 2]); %[Readout Lines Coils]
kspace = reshape(kspace, [twix_obj.image.NCol ,twix_obj.image.NLin/twix_obj.image.NSeg, twix_obj.image.NSeg, twix_obj.image.NCha]);
disp('Size of reshaped k-space: ')
disp(size(kspace))

%% k-space sampling
flagSelfNav = 1; % Was SI-projections acquired?
flagPlot = 1; % Plot example shot
[kx, ky, kz] = computePhyllotaxis(twix_obj.image.NCol, twix_obj.image.NLin/twix_obj.image.NSeg, twix_obj.image.NSeg, flagSelfNav, flagPlot);



