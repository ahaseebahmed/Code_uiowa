function [raw, header, rawstruct, raw_read_status ] = read_MR_rawdata(in_name, save_baseline, phaselist, echolist, slicelist, rcvrlist)
%----------------------------------------------------------------
% function [raw, header, rawstruct] = read_MR_rawdata(in_name, save_baseline, phaselist, echolist, slicelist, rcvrlist)
% Revised by L. Sacolick 10-12-2012: added recon case for 2dptxcal pfiles.
%


raw_read_status = 0;
ec=0;
non_standard_pfile=0;

% Determine if the input parameter IN_NAME points to a directory of a file
%[ dirname, filename, fileext, fileversn] = fileparts(in_name);
%filename = [filename fileext fileversn] ;
%   CAC 171012, removed 'fileversn' from check, was giving error:
%       Error using fileparts
%       Too many output arguments.
[ dirname, filename, fileext] = fileparts(in_name);
filename = [filename fileext] ;

[rows num_char]=size(filename);
P_check=strmatch('P',filename);
ext_check=strmatch('.7',fileext,'exact');
if(num_char ~= 8 | P_check ~= 1 | ext_check ~= 1)
    non_standard_pfile=1;
end

fullfilename = fullfile( dirname, filename);

%header = read_MR_headers( fullfilename, 'all');
%%% Get only required headers, CAC 171214
% Check p-file version and use orchestra if necessary
%   CAC 180718
try
    header_rdb = read_MR_headers( fullfilename, 'rdb');
catch exception_rdb
    header_rdb = read_MR_headers( fullfilename, 'orchestra');
end
if header_rdb.rdb_hdr.rdbm_rev <= 24.0
    header = read_MR_headers( fullfilename, 'cac_hack');
else
    header = read_MR_headers( fullfilename, 'orchestra');
end



rawstruct.basename = fullfilename;
rawstruct.data     = struct('filename','');
% Get endian from header structure
endianID = header.endian;
% Pull useful info from rdb_hdr
nfphases   = header.image.fphase;
psdname    = deblank(header.image.psd_iname);
npasses    = header.rdb_hdr.npasses;
da_xres    = header.rdb_hdr.da_xres;
da_yres    = header.rdb_hdr.da_yres ;

nslices    = header.rdb_hdr.nslices;
nechoes    = header.rdb_hdr.nechoes;
point_size = header.rdb_hdr.point_size;
receivers = (header.rdb_hdr.dab(2)-header.rdb_hdr.dab(1))+1;
nslices_per_pass = sum( header.data_acq_tab.pass_number == 0 );  % Number of slices in each pass

switch point_size
    case 2
        raw_data_type = 'int16';       % 16 bit data
    case 4
        raw_data_type = 'int32';       % 32 bit (EDR)
end

if(nfphases == 0)
    nphases = 1;
else
    nphases = nfphases;
end

nphases = 1 ;

% Unless passed at least 6 input parameters, read ALL the data in the rawfile
phaselist = 1:nphases;
echolist  = 1:nechoes;
slicelist = 1:nslices;
rcvrlist  = 1:receivers;
save_baseline = 'db';

% Compute (byte) size of frames, echoes and slices
elements  = da_xres*2;
frame_sz  = 2 * da_xres * point_size;     % Size of one frame of rawdata (one Kx line)
echo_sz   = frame_sz * da_yres;           % Size of one image (one Kx * Ky matrix)
slice_sz  = echo_sz  * nechoes;           % Size of nechoes images (one raw matrix * number of echoes)
mslice_sz = slice_sz * receivers;	      % (one slice * number of receivers)

% Determine which of the 3 types of temporal phase raw data it is
temporal_type='na';

% Compute offset in bytes to start of raw_data, and the act_yres to use.
if(strcmp(save_baseline,'sb'))
    baseline_offset=0;
    act_yres=da_yres;
elseif(strcmp(save_baseline,'db'))
    baseline_offset   = da_xres * 2 * point_size;
    act_yres=da_yres-1;
else
    errordlg('You must specify sb or db for baseline argument.','ERROR!','modal');
    raw=[];
    ec=4;
    raw_read_status = -1;
    return;
end

data_elements = 2 * da_xres * act_yres;
total_bytes = header.total_length + nslices_per_pass * mslice_sz;
raw_file_skip=1;

%SACOLICK added case for 2dptxcal:
try
    if(nslices_per_pass > 1)
        raw_file_skip = nslices_per_pass;
    end
catch
end

% Check for complete raw data set: all files present and of the expected length
%  fprintf('\nRawfile Dataset information:\n');
%  fprintf('  File format:            %10s\n', header.format );
%  fprintf('  Endian ID:              %10s\n', header.endian );
%  fprintf('  Header size, bytes:     %10d\n', header.total_length );
% fprintf('  Slice size, bytes:      %10d\n', mslice_sz );
%  fprintf('  Number of Time Phases:  %10d\n', nphases);
%  fprintf('  Expected Pfile, bytes:  %10d\n', total_bytes );
%  fprintf('Check rawfiles Status:\n');
file_inc=raw_file_skip;
for filei = 1:npasses
    filename = sprintf('P%05d.7', header.rdb_hdr.run_int+file_inc-raw_file_skip) ;
    file_inc=file_inc+raw_file_skip;
    fidi = fopen( fullfile( dirname, filename), 'r', endianID );
    if fidi < 3
        status_str = 'Could not be opened\n';
        raw_read_status = -1;
        fprintf( '  Pass %2d rawfile:%12s   %s\n', filei, filename, status_str);
        return;
    else
        fseek( fidi, 0, 'eof' );
        filesize = ftell( fidi );
        fclose( fidi );
        if filesize~=total_bytes
            status_str = sprintf('Wrong rawfile size: %d bytes (actual) \n', filesize);
            raw_read_status = -1;
            fprintf( '  Pass %2d rawfile:%12s   %s\n', filei, filename, status_str);
            return;
        else
            status_str = 'OK';
        end
    end
    %fprintf( '  Pass %2d rawfile:%12s   %s\n', filei, filename, status_str);
    rawstruct.data(filei).filename = filename;
end

%  fprintf('  Echo Slice Rcvr Pass Filename     Offset   Status\n');
% fprintf('  -------------------------------------------------\n');
act_yres;
max(phaselist);
max(echolist);
max(slicelist);
max(rcvrlist);

raw = zeros( act_yres, da_xres, max(phaselist),max(echolist), max(slicelist), max(rcvrlist) );
for echonum = min(echolist):max(echolist)
    echo = echolist(echonum - min(echolist) + 1);
    echo_offset = echo_sz * (echo-1);
    for slicenum = min(slicelist):max(slicelist)
        slice_requested = slicelist(slicenum - min(slicelist) + 1);
        pass  = header.data_acq_tab.pass_number(slice_requested);
        slice_offset = slice_sz * (header.data_acq_tab.slice_in_pass(slice_requested)-1);
        for rcvrnum = min(rcvrlist):max(rcvrlist)
            rcvr = rcvrlist(rcvrnum - min(rcvrlist) + 1);
            receiver_offset  = nslices_per_pass * slice_sz * (rcvr-1);

            % fprintf('%10d %10d %10d %10d %10d\n', header.total_length, receiver_offset, slice_offset, echo_offset, baseline_offset);
            file_offset       = header.total_length + receiver_offset + slice_offset + echo_offset + baseline_offset;

            if(non_standard_pfile == 0)
                if(raw_file_skip > 1)
                    filename = sprintf('P%05d.7', header.rdb_hdr.run_int+header.data_acq_tab.pass_number(slice_requested)*raw_file_skip);
                else
                    filename = sprintf('P%05d.7', header.rdb_hdr.run_int+header.data_acq_tab.pass_number(slice_requested));
                end
                fid = fopen( fullfile( dirname, filename), 'r', endianID);

            else
                fid = fopen( fullfilename, 'r', endianID);
            end
            if fid<3
                status_str = 'File not found';
                raw_read_status = -1;
                disp( status_str);
            else
                status = fseek(fid, file_offset, 'bof');
                if status==-1
                    status_str = 'Offset outside of file';
                    raw_read_status = -1;
                    disp( status_str);
                else
                    [raw_data, count] = fread( fid, 2 * da_xres * act_yres, raw_data_type);
                    if count~=(2*da_xres*act_yres)
                        status_str = 'Could not read sufficient data';
                        raw_read_status = -1;
                        disp( status_str);
                        return;
                    else
                        status_str = 'OK';
                        raw_data = complex( raw_data(1:2:count), raw_data(2:2:count) );
                        raw_data = reshape( raw_data, da_xres, act_yres );
                        raw(:,:,1,echonum, slicenum, rcvrnum) = transpose( raw_data );
                    end
                end
                fclose(fid);
            end
            %       fprintf(' %4d %4d %4d %4d %10s %10d   %s\n', ...
            %         echo, slice_requested, rcvr, pass, filename, file_offset, status_str);
        end
    end
end


return
