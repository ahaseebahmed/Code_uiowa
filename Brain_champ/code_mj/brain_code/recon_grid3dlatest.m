function [bb,bbabs] = recon_grid3d1(d,h,wfn,mtx_reco,delay,lbt,spks,...
    fname,f0,do_single,verb)
%RECON_GRID3D Reconstruct single-shot spiral trajectory
%[bb,bbabs] = recon_grid3d(d,h,wfn,mtx_reco,delay,lbt,spks,fname,f0,...
%                          do_single,verb);
%                                                          [unit] (default)
%         d  Raw data (or P-File/ScanArcihive fname)
%         h  Header from p-file (or empty)
%       wfn  Location of waveform .mat file (or wf structure)
%       mtx  Reconstruction matrix resolution (Fourier interp.)   ([])
%     delay  Gradient-acquisition delay                    [us]   (0)
%            1d or 2d=[WHOLE,ZOOM]
%       lbt  Gaussian linebroadening time                  [Hz]   (0)
%      spks  Remove spike noise                                   (false)
%     fname  Print <fname>.png and save reco as <fname>.mat       ([]) 
%            also export dicom if template.dcm is found
%        f0  Change in centre frequency                    [Hz]   (0)
% do_single  Reconstruct/store data in single precision           (true)
%      verb  Verbose mode (0=off; 1=printf; 2=printf+plotting     (2)
%
%        bb  Reconstructed data     (mtx,mtx,mtx,#timesteps,#coils)
%     bbabs  RMS coil-combined data (mtx,mtx,mtx,#timesteps)
%
% 7/2019  Rolf Schulte
if (nargin<1), help(mfilename); return; end


%% input variables
RFS_MAX_NSCANS = 16382;                % maximum number of scans
if ~exist('mtx_reco','var'), mtx_reco = []; end
if isempty(mtx_reco),        mtx_reco = 0; end
if ~exist('delay','var'), delay = []; end
if isempty(delay),        delay = 0; end
if ~exist('lbt','var'),   lbt = []; end
if isempty(lbt),          lbt = 0; end
if ~exist('spks','var'),  spks = []; end
if isempty(spks),         spks = false; end
if ~exist('fname','var'), fname = []; end
if ~isempty(fname)
    if ~islogical(fname)
        if ~isempty(regexpi(fname,'\.7$')), fname = fname(1:end-2); end
        if ~isempty(regexpi(fname,'\.h5$')), fname = fname(1:end-3); end
        if ~isempty(regexpi(fname,'\.mat$')), fname = fname(1:end-4); end
    end
end
if ~exist('f0','var'),    f0 = []; end
if isempty(f0),           f0 = 0; end
if length(f0)~=1, error('length(f0)~=1'); end
if ~exist('do_single','var'), do_single = []; end
if isempty(do_single),    do_single = true; end
if length(do_single)~=1,  error('length(do_single)~=1'); end
if ~exist('verb','var'),  verb = []; end
if isempty(verb),         verb = 2; end
if verb>0, timerVal = tic; end          % record execution time


%% reading in data, if pfile name is given
if ~isnumeric(d)
    if exist(d,'file')
        [d,h] = read_p(d);
    else
        warning('strange input d/file not existing');
    end
end
%% reading waveform file
if isempty(wfn), error('wfn empty'); end
if isstruct(wfn)
    wf = wfn;
else
    if ~isempty(regexpi(wfn,'\.wav$')), wfn = wfn(1:end-4); end
    if isempty(regexpi(wfn,'\.mat$')),  wfn = [wfn '.mat']; end
    if ~exist(wfn,'file'), error('file not found: wfn=%s',wfn); end
    wf = load(wfn);          % load waveform
end


%% different WHOLE and ZOOM delay times (delay->array)
if length(delay)>1
    switch h.rdb_hdr.grad_mode
        case 1, delay = delay(1);
        case 2, delay = delay(2);
        otherwise
            error('length(delay)(=%g)>1 & h.rdb_hdr.grad_mode(=%g) ~= 1 or 2',...
                length(delay),h.rdb_hdr.grad_mode);
    end
end


%% check for fields required for reconstruction
if ~isfield(wf,'mtx'), error('wf.mtx not existing'); end
if isfield(wf,'nviews')
    % reduced mat size: only single view stored; regenerate info
    if ~isfield(wf,'k')
        if ~isfield(wf,'ks') && ~isfield(wf,'phi') && ~isfield(wf,'theta')
            error('wf.ks,phi,theta not existing');
        else
            ks = squeeze(wf.ks);
            nks = size(ks,1);
            if do_single, ks = single(ks); end
            k = zeros(nks,wf.nviews,3,class(ks));
            
            sinphi = sind(wf.phi);
            cosphi = cosd(wf.phi);
            sintheta = sind(wf.theta);
            costheta = cosd(wf.theta);
            
            k(:,:,1) =  ks(:,1)*(costheta.*cosphi) + ...
                ks(:,2)*(sinphi) - ...
                ks(:,3)*(sintheta.*cosphi);
            k(:,:,2) =  -ks(:,1)*(costheta.*sinphi) + ...
                ks(:,2)*(cosphi) + ...
                ks(:,3)*(sintheta.*sinphi);
            k(:,:,3) = ks(:,1)*(sintheta) + ...
                ks(:,3)*(costheta);
            % k = permute(k,[2 1 3]);
            k = reshape(k,[1 nks*wf.nviews 3]);
        end
    else
        k = wf.k;
    end
    if ~isfield(wf,'dcf')
        if ~isfield(wf,'dcfs')
            wfn_dcf = [wfn(1:end-4) '_dcf.mat'];
            if exist(wfn_dcf,'file')
                if verb>0
                    fprintf('Loading dcf from file ''%s''\n',wfn_dcf);
                end
                load(wfn_dcf);
            else
                if verb>0
                    fprintf('Calculating dcf via calc_sdc3\n');
                    fprintf('\tand storing to ''%s''\n',wfn_dcf);
                end
                dcf = calc_sdc3(k,wf.mtx,5);
                save(wfn_dcf,'dcf');
            end
        else
            dcfs = wf.dcfs(:).';
            if do_single, dcfs = single(dcfs); end
            dcf = repmat(dcfs,[1 wf.nviews]);
        end
    else
        dcf = wf.dcf;
        if isa(dcf,'uint16')
            if do_single
                dcf = single(dcf)/(2^16-1)*wf.dcf_range;
            else
                dcf = double(dcf)/(2^16-1)*wf.dcf_range;
            end
        end
    end
    if ~isfield(wf,'ind')
        if ~isfield(wf,'inds')
            error('wf.inds not existing'); 
        else
            ind = repmat(wf.inds,[wf.nviews 1]);
        end
    else
        ind = wf.ind;
    end
    if ~isfield(wf,'t')
        if ~isfield(wf,'ts')
            error('wf.ts not existing'); 
        else
            ts = wf.ts(:).';
            if do_single, ts = single(ts); end
            t = repmat(ts,[1 wf.nviews]);
        end
    else
        t = wf.t;
        t = t.'; t = t(:).';
    end
else
    if ~isfield(wf,'k'),   error('wf.k not existing'); end
    if ~isfield(wf,'dcf'), error('wf.dcf not existing'); end
    if ~isfield(wf,'ind'), error('wf.ind not existing'); end
    if ~isfield(wf,'t'),   error('wf.t not existing'); end
    k = wf.k;
    dcf = wf.dcf;
    ind = wf.ind;
    t = wf.t;
    if size(t,1)==1, t = repmat(t,[size(ind,1) 1]); end
    t = t.'; t = t(:).';
end

if isempty(dcf),    error('dcf is empty'); end
if any(isnan(dcf)), error('dcf contains NaN'); end
if any(isinf(dcf)), error('dcf contains inf'); end

mtx_acq = wf.mtx(1);         % nominal matrix resolution of acquisition
clear wf

if mtx_reco <= 0, mtx_reco = mtx_acq; end
k = k*mtx_acq/mtx_reco;      % scale k-space -> Fourier interpolation
fov = h.rdb_hdr.fov*1d-3;    % actual field-of-view [m]

if do_single
    k = single(k);
    dcf = single(dcf);
    t = single(t);
else
    k = double(k);
    dcf = double(dcf);
    t = double(t);
end


%% reshape data for vap_continue_loop
if ((sum(ind(:))>RFS_MAX_NSCANS) && (size(d,4)>1))
    if verb>0, fprintf('Reshaping data for vap_continue_loop\n'); end
    [n1,n2,n3,n4,n5,n6] = size(d);
    tmp = complex(zeros(n1*n4,n2,n3,1,n5,n6,class(d)));
    for l4=1:n4
        ii = (1:n1)+(l4-1)*n1;
        tmp(ii,:,:,1,:,:) = d(:,:,:,l4,:,:);
    end
    d = tmp;
    clear tmp
end


%% misc variables to checkings
bw = h.rdb_hdr.user0;
% n_exc = h.rdb_hdr.user4;
% n_slice = size(d,5);
[nexc,~,n3,~,~,ncoils] = size(d);
xloc = h.rdb_hdr.user26;
yloc = h.rdb_hdr.user27;
zloc = h.rdb_hdr.user28;
rot90fac = h.data_acq_tab.rotate;
trnsps = h.data_acq_tab.transpose;
if ~((trnsps(1)==0)||(trnsps(1)==3))
    warning('h.data_acq_tab.transpose (=%g) not 0 or 3: unknown image orientation',trnsps(1));
end
if n3~=1, warning('size(d,3)(=%g)~=1',n3); end
if nexc<size(ind,1)
    warning('nexc(=%g)<size(ind,1)(=%g)',nexc,size(ind,1));
end
tmp = nexc/size(ind,1);
if abs(tmp-floor(tmp))>1d-10
    warning('nexc(=%g) not multiple of size(ind,1)(=%g); truncating data',...
        nexc,size(ind,1));
    d = d(1:floor(tmp)*size(ind),:,:,:,:,:);
    % nexc = size(d,1);
end


%% pre-processing of data
if spks
    if verb>0, fprintf('Removing spike noise\n'); end
    d = remove_spikes(d,[],false); 
end
if ((abs(xloc)>0.01) || (abs(yloc)>0.01) || (abs(zloc)>0.01))
    shft = mtx_reco*[xloc yloc zloc]*1d-3/fov;
else
    shft = [];
end

cart_down_fac = []; 
if do_single
    dd = raw2grid(single(d),ischop(h),k,shft,cart_down_fac,[],ind,delay*1d-6,bw,false);
else
    dd = raw2grid(d,ischop(h),k,shft,cart_down_fac,[],ind,delay*1d-6,bw,false);
end
clear d wf ind
% size of reshaped data dd:
%  dim1=#coils; dim2=#indexed kspace data; dim3=#excitations; dim4=#slices
if lbt>0, dd = ak_apodise(dd,[],[],t,lbt,false); end
if f0~=0, phafu = exp(1i*2*pi*t*f0); end
clear t


%% allocate memory for reconstructed images
ntimesteps = size(dd,3);
if do_single
    if verb>0, fprintf('Reconstructing/storing data as single\n'); end
    bb = complex(zeros(mtx_reco,mtx_reco,mtx_reco,ntimesteps,ncoils,'single'));
else
    bb = complex(zeros(mtx_reco,mtx_reco,mtx_reco,ntimesteps,ncoils));
end


%% actual reconstruction
for lt=1:ntimesteps
    if do_single
         if f0==0
             b1 = gridding3dsingle(k,dd(:,:,lt),dcf,[1 1 1]*mtx_reco);
         else
             b1 = gridding3dsingle(k,bsxfun(@times,dd(:,:,lt),phafu),dcf,[1 1 1]*mtx_reco);
         end
         
         for lc=1:ncoils
             % image orientation
             for l3=1:mtx_reco
                 if trnsps(1)==3
                     bb(:,:,l3,lt,lc) = rot90(fliplr(b1(:,:,l3,lc)).',rot90fac(1));
                 else
                     bb(:,:,l3,lt,lc) = rot90(fliplr(b1(:,:,l3,lc)),rot90fac(1));
                 end
             end
         end
    else
        for lc=1:ncoils
            if f0==0
                b1 = gridding3d(k,dd(lc,:,lt),dcf,[1 1 1]*mtx_reco);
            else
                b1 = gridding3d(k,dd(lc,:,lt).*phafu,dcf,[1 1 1]*mtx_reco);
            end
            
            % image orientation
            matver = version;
            if str2double(matver(1:3))>8.2
                if trnsps(1)==3
                    bb(:,:,:,lt,lc) = rot90(permute(fliplr(b1),[2 1 3]),rot90fac(1));
                else
                    bb(:,:,:,lt,lc) = rot90(fliplr(b1),rot90fac(1));
                end
            else
                for l3=1:mtx_reco
                    if trnsps(1)==3
                        bb(:,:,l3,lt,lc) = rot90(fliplr(b1(:,:,l3)).',rot90fac(1));
                    else
                        bb(:,:,l3,lt,lc) = rot90(fliplr(b1(:,:,l3)),rot90fac(1));
                    end
                end
            end
        end
    end
end


%% coil combine
if ncoils>1
    bbabs = abs(sqrt(sum(bb.*conj(bb),5)));
else
    bbabs = abs(bb);
end


%% saving result as mat file
if ~isempty(fname)
    bb = single(bb);   % convert from double (64bit) to single (32bit) precision
    save([fname '.mat'],'-mat','-v7.3',...
        'bb','h','wfn','mtx_reco','delay','lbt','spks','fname',...
        'cart_down_fac');
end


%% plotting: 3 orientations
if verb>1
    figure;
    imagesc_ind3d(bbabs,'','','',false,true);
    colormap gray;
    figstr = sprintf('P%05d Exam%d Series%d',...
        h.image.rawrunnum,h.exam.ex_no,h.series.se_no);
    
    set(gcf,'name',figstr);
    drawnow
    
    if ~isempty(fname)
        print([fname '.png'],'-dpng','-r600','-painters');
    end
end


%% plotting: slices + timesteps
if verb>2
    for lt=1:ntimesteps
        figure;
        % imagesc_row(bbabs,[0 1],'ind',true,true);
        imagesc_row(bbabs(:,:,:,lt),[],'',true,true);
        figstr = sprintf('P%05d Exam%d Series%d: ',...
            h.image.rawrunnum,h.exam.ex_no,h.series.se_no);
        if ntimesteps>1, figstr = sprintf('%stimestep=%g',figstr,lt); end
        set(gcf,'name',figstr);
        colormap(gray);
        if ~isempty(fname)
            print('-dpng',[fname '_lt' num2str(lt)]);
        end
    end
end


%% exporting images as dicom into scanner database
if ~isempty(fname)
    % export as dicom, if fname present fname_template.dcm is found
    template_fname = [fname '_template.dcm'];
    if ~exist(template_fname,'file')
        if verb>0
            fprintf('No template dicom file found\n\t%s\n',...
                template_fname);
        end
    else
        if verb>0, fprintf('Storing dicom images\n'); end
        inp.SeriesNumber = h.exam.ex_numseries;
        inp.SeriesDescription  = sprintf('MNSRP;3D%g;TS%g;%s',...
            mtx_reco,ntimesteps,h.series.se_desc);
        inp.MRAcquisitionType = '3D';
        inp.SliceThickness = h.image.slthick;
        inp.RepetitionTime = round(h.image.tr*1d-3);
        inp.EchoTime = round(h.rdb_hdr.user25*1d-1)*1d-2;
        write_dicom(fname,bbabs,template_fname,inp);
    end
end


if verb>0
    fprintf('recon_grid3d.m finished: ');
    fprintf('total reconstruction time = %.2f [s]\n',toc(timerVal));
end
if nargout<1, clear bb; end
    
end      % main function recon_grid3d.m
