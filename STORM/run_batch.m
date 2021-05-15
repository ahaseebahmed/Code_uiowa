clear all;
addpath('gspbox');
gsp_start;
addpath('MatlabFiles');

%% Initialization of GPU and Matlabpool
% replace with gpuDevice() ?

A = gpuArray(10);
clear A;

poolobj = gcp('nocreate');
if isempty(poolobj)
    parpool('local',10);
end

%% Input and output parameters

RawDataDirectory = './../../../Storm_Basis/STORM/RawData/';
ReconDataDirectory = './../../ReconData/';


%% Trajectory parameters

param.common_lines = 4; % Number of navigator lines per frame
param.nf = 200;
param.lines_per_SG_block = 10; % Number of navigator lines + Number of Golden Angle lines per frame
param.T = 0.05;

param.blocks_to_skip = 100; % Number of frames; initial data is often corrupted

%% Optimization parameters
param.CoilSolveCGIterations = 50;

param.verbosity = 2;
param.LaplacianRegParam = 1e-1; % Higher for more dense Laplacian

param.NcoilSelect = 20;
param.useGPU = 1; %  GPU processing


%% Enable for L2 regularization
%
% param.SolverType = 'L2'
% param.CoeffsCGIterations = 100;
% param.lambdaCoeffs = 0.2;


%% Enable for L1 regularization
param.SolverType = 'L1';
param.CoeffsCGIterations = 15;
param.lambdaCoeffs = 5e-5;
param.eps = 0.025;
param.outerIterations = 4;


%% Process data one by one

liste = rdir([RawDataDirectory,'**/*SG*.dat'],'',RawDataDirectory);
files = {liste.name};
param.filename = [RawDataDirectory,files{1}];


for fileindex=1:length(files)
    
    param.filename = [RawDataDirectory,files{fileindex}];
    newdirname = strtok(files{fileindex},'/');
    
    
    
    if(~exist([ReconDataDirectory,newdirname],'dir'))
        mkdir([ReconDataDirectory,newdirname]);
    end
    
    outname = [ReconDataDirectory,strtok(files{fileindex},'.'),'_',param.SolverType,'_lam_',num2str(param.lambdaCoeffs),'.mat'];
    
    if(~exist(outname,'file'))
      
        
        fprintf('Processing %s\n',param.filename);
        fprintf('----------------------------\n');
    
        twix_obj = mapVBVD(param.filename);
        lines_to_skip  = param.blocks_to_skip*param.lines_per_SG_block; 

        kSpaceData = twix_obj{2}.image();
        
        clear twix_obj;
    
        
        [U1,D,csm,coilimages,paramOut] = storm_do_all(param,kSpaceData);
    
        fprintf('Saving %s..\n',outname);
        save(outname ,'U1', 'D', 'paramOut', 'csm', 'coilimages');
    else
        fprintf('Recon file %s already exists; exiting \n',outname);
        fprintf('-------------------------------------\n');
    end

end