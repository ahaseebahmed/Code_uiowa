clear all;
%addpath('gspbox');
%gsp_start;
addpath('MatlabFiles');

%% Initialization of GPU and Matlabpool
% replace with gpuDevice() ?

%temp
A = gpuArray(10);
clear A;

% poolobj = gcp('nocreate');
% if isempty(poolobj)
%     parpool('local',10);
% end

%% Input and output parameters

RawDataDirectory = '/Shared/lss_jcb/abdul/prashant_cardiac_data/others/';
ReconDataDirectory = '/Shared/lss_jcb/abdul/prashant_cardiac_data';


%% Trajectory parameters

param.common_lines = 2; % Number of navigator lines per frame
param.nf = 900;
param.lines_per_SG_block = 10; % Number of navigator lines + Number of Golden Angle lines per frame
param.T = 0.05;

param.blocks_to_skip = 100; % Number of frames; initial data is often corrupted

%% Optimization parameters
param.CoilSolveCGIterations = 50;

param.verbosity = 2;
param.LaplacianRegParam = 1e-1; % Higher for more dense Laplacian

param.NcoilSelect = 38;
param.useGPU = 1; %  GPU processing
param.Nbasis = 30;
sl_to_pick=[3];%[2,3,4,7,8,9,10,11,12];
%% Enable for L2 regularization
%
 param.SolverType = 'L2'
 param.CoeffsCGIterations = 80;
 param.lambdaCoeffs = 0.1;
 param.szSmooth = 13;

%% Enable for L1 regularization
% param.SolverType = 'L1';
% param.CoeffsCGIterations = 15;
% param.lambdaCoeffs = 5e-5;
% param.eps = 0.025;
% param.outerIterations = 4;


%% Process data one by one

liste = rdir([RawDataDirectory,'*039_SAX_gNAV_gre.dat'],'',RawDataDirectory);
files = {liste.name};
param.filename = [RawDataDirectory,files{1}];


for fileindex=1:length(files)
    
    param.filename = [RawDataDirectory,files{fileindex}];
    newdirname = strtok(files{fileindex},'/');
    
    
    
%     if(~exist([ReconDataDirectory,newdirname],'dir'))
%         mkdir([ReconDataDirectory,newdirname]);
%     end
    
    
   % if(~exist(outname,'file'))
      
        
        fprintf('Processing %s\n',param.filename);
        fprintf('----------------------------\n');
    
        twix_obj = mapVBVD(param.filename);
        lines_to_skip  = param.blocks_to_skip*param.lines_per_SG_block; 

        kSpaceData_slc = twix_obj{2}.image();
        
        clear twix_obj;
    for i=1:length(sl_to_pick)%size(kSpaceData_slc,5)
        cl=[3,4,6,7,8,13,14,17,18,19,26,34];
        [kdata] = storm_do_all1(param,squeeze(kSpaceData_slc(:,cl,:,1,sl_to_pick(i))));
        %[U1,D,csm,coilimages,paramOut,Ul2] = storm_do_all(param,squeeze(kSpaceData_slc(:,:,:,1,i)));
%         S_mask=zeros(512*512,900);
%         for jj=1:900 
%             S_mask(S{jj},jj)=1;
%         end
%         S_mask=reshape(S_mask,512,512,900);
%         
%         tmp = (zeros(512^2,1));
%         kdata=zeros(512^2,900,size(csm,3));
%         for ij=1:900  
%             for kj=1:size(b,2)
%                 tmp(S{ij})= b{ij,kj};
%                 kdata(:,ij,kj)=tmp;
%                 tmp(:)=0;
%             end
%         end
        
        %outname = [ReconDataDirectory,newdirname,'/',strtok(files{fileindex},'.'),'_',param.SolverType,'_lam_',num2str(i),'.mat'];
        kdata=reshape(kdata,[512,3,10,900]);
        outname = [ReconDataDirectory,'/',strtok(files{fileindex},'.'),num2str(sl_to_pick(i)),'_kd.mat'];
        
        fprintf('Saving %s..\n',outname);
        %save(outname ,'U1', 'D', 'paramOut', 'csm', 'coilimages');
        save(outname ,'kdata','-v7');
    end
    %else
     %   fprintf('Recon file %s already exists; exiting \n',outname);
      %  fprintf('-------------------------------------\n');
    %end

end