
% uigetpfile3.m
%   Load a pfile header and raw data from dialog
%   Curt Corum, 200801

startDir = pwd;

FilterSpec = 'P*.*';
[FileName, PathName] = uigetfile( FilterSpec);
FilePath = fullfile( PathName, FileName);
fprintf( 'Reading p-file header for %s\n', FilePath);

[DirPath, FName, FExt] = fileparts( FilePath);
FName = strcat( FName, FExt);
cd( DirPath);

pfileHandle = GERecon( 'Pfile.Load', FName, 'Full-Anonymize');
GERecon( 'Pfile.SetActive', pfileHandle);
header.orchestra_header = GERecon( 'Pfile.Header');

% construct old header fields from ochestra fields
header.rdb_hdr = header.orchestra_header.RawHeader;
header.data_acq_tab = header.orchestra_header.DataAcqTable;
header.image = header.orchestra_header.ImageData;
        
% get the raw data
fprintf( 'Reading p-file raw data from %s\r', FName);
kSpace = GERecon('Pfile.KSpace', 1, 1);
view = GERecon('Pfile.ViewData', 1, 1);
cd( startDir);