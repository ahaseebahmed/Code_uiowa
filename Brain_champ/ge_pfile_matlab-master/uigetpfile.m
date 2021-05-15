
% uigetpfile.m
%   Load a pfile from dialog
%   Curt Corum, 171304

FilterSpec = 'P*.*';
[FileName, PathName] = uigetfile(FilterSpec);
FilePath = fullfile(PathName, FileName);
fprintf('%s\r', FilePath);

%fileID = fopen(FilePath, 'r', 'l');
%RawData = fread(fileID,40000,'float', 40000);
%plot(RawData);
%a = read_rdb_hdr( fileID,20.006 );

%[header, read_hdr_status ]  = read_MR_headers( FilePath, 'all');
%[header, read_hdr_status ]  = read_MR_headers( FilePath, 'cac_hack');
[header, read_hdr_status ]  = read_MR_headers( FilePath, 'rdb');
%[header, read_hdr_status ] = read_MR_headers( FilePath, 'orchestra');

%[data,header] = read_p(FilePath);

[raw, header, rawstruct, raw_read_status ] = read_MR_rawdata( FilePath);

%%% does not work for Silent Scan, orchestra 1.6.1
%pfileHandle = GERecon('Pfile.Load', FilePath);
%pfileHandle = GERecon('Pfile.Header', FilePath);
%GERecon('Pfile.SetActive', pfileHandle);
%header = GERecon('Pfile.Header', pfileHandle)

