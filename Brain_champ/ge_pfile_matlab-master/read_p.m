function [data,header] = read_p(fname,correction)
% READ_P Read a raw data GE MR P-file.
%  [data,header] = read_p(fname,correction)
%     fname  filename (if missing -> gui)
%correction  apply corrections (default=[] -> determine from header)
%            0  none
%            1  ptx: if ((header.rdb_hdr.user2==13) && (f0>35d6) && 
%                        (header.rdb_hdr.rdbm_rev<15))
%               correct for updown convert: f0=160d6-f0 & data=conj(data)
%            2  mr901: if
%               f0=f0+1931067306
% 
% 7/2015  Rolf Schulte
% See also read_MR_rawdata, read_MR_headers.

%% default input
if ~exist('fname','var'),       fname = []; end
if ~exist('corrections','var'), correction = []; end
f0corrmr901 = 193106730.6;        % correction freq for MR901


%% open gui if fname empty
if isempty(fname),
    [fname,pname] = uigetfile({'*.*'},'Select P-file');
    fname = [pname fname];
end


%% check if file found
if ~exist(fname,'file'), error('file %s not found',fname); end


%% enable gunzip for Pxxx.7.gz
removefile = false;
if ~isempty(regexpi(fname,'\.gz$')),
    if exist(fname(1:end-3),'file'), 
        error('gunzip-ped file %s already existing',fname); 
    end
    gunzip(fname);
    removefile = true;
    oldfname = fname;
    fname = fname(1:end-3);
end


%% actual file reading via GE reading routines
[data,header] = read_MR_rawdata(fname);
if removefile,
    if exist(oldfname,'file'), delete(fname); 
    else error('old file (%s) not existing anymore',oldfname);
    end
end


%% check headers if correction necessary
if isempty(correction),
    f0 = header.rdb_hdr.ps_mps_freq/10;
    correction = 0;
    % ptx
    if ((header.rdb_hdr.user2==13) && (f0>35d6) && (header.rdb_hdr.rdbm_rev<15)),
        correction = 1;
    end
    % MR901
    if ((f0<0) && (round(header.rdb_hdr.rdbm_rev*10)==144)),
        correction = 2;
    end
end


%% apply miscelaneous correcions
switch correction
    case 0, 
    case 1,        % PTX updown converter for 13C
        fprintf('\nAttention: up-down-converter: Nuc=13,f0=%d\n',round(f0));
        fprintf('\t-> f0=160d6-f0  and  data=conj(data)\n\n');
        header.rdb_hdr.ps_mps_freq = (160d6 - f0)*10;
        data = conj(data);
    case 2,        % MR901
        fprintf('\nAttention: MR901; correcting f0=f0(=%g)+f0cor(=%g)=%g\n\n',...
            round(f0),round(f0corrmr901),round(f0+f0corrmr901));        
        header.rdb_hdr.ps_mps_freq = (f0+f0corrmr901)*10;
    otherwise, fprintf('Warning: correction(=%g) not found\n',correction);
end
