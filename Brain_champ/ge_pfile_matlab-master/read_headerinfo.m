%%%%%%% read text file containing header info %%%
function hdr_szs = read_headerinfo(textfile,rdbm_rev)
%Assumes the following format
% #REVNUM
% headersize
% offset1
% offset2
% offset3
% ...
% #end
% As long as the values between the #REVNUM and #end are the correct values
% and no corruption occurs there, the reader does not care about anything
% else outside

fid = fopen(textfile,'r');
flag = 0;
cnt = 1;
hdr_szs(1) = -1;
while ~feof(fid)
    str = fgetl(fid);
    if isempty(str)
        continue
    end
    keyString = str(1:min(2,length(str)));
    if strcmp(keyString(1), '#') && ~strcmp(keyString, '#e')
        % found a line with #REVNUM ... read it
        if str2double(str(2:length(str))) == rdbm_rev
            % found the rev that I was looking for
            str = str(2:length(str));
            flag = 1;
            cnt = 1;
        end
    elseif strcmp(keyString, '#e')
        if flag == 1
            %done with reading header offsets
            return;
        end
        %otherwise just continue dont do anything
    end
    if flag == 1
        hdr_szs(cnt) = str2double(str);
        cnt = cnt+1;
    end
end