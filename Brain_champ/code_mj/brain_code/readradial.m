function [dd,k,dcf,mtx_acq,nviews,fov,phi,theta] = readradial(d,h,wfn,mtx_reco,delay,lbt,spks,...
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
nviews = wf.nviews;
fov = wf.fov1H;
phi = wf.phi;
theta = wf.theta;
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

%% compress data for memory issues

%% reshape data for vap_continue_loop
if ((sum(ind(:))>RFS_MAX_NSCANS) && (size(d,4)>1))
    if verb>0, fprintf('Reshaping data for vap_continue_loop\n'); end
    [n1,n2,n3,n4,n5,n6] = size(d);
    if(do_single)
        fprintf('Converting to single \n');
        d = single(d);
        fprintf('Convertion done \n');
    end
    tmp = complex(zeros(n1*n4,n2,n3,1,n5,n6,class(d)));
    for l4=1:n4
        fprintf('Unrolling %d \n',l4);
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
 fprintf('Starting raw2grid %d \n',l4);
if do_single
    dd = raw2grid(d,ischop(h),k,shft,cart_down_fac,[],ind,delay*1d-6,bw,false);
else
    dd = raw2grid(d,ischop(h),k,shft,cart_down_fac,[],ind,delay*1d-6,bw,false);
end
 fprintf('Raw2grid done %d \n',l4);

clear d wf ind
% size of reshaped data dd:
%  dim1=#coils; dim2=#indexed kspace data; dim3=#excitations; dim4=#slices
if lbt>0, dd = ak_apodise(dd,[],[],t,lbt,false); end
%if f0~=0, phafu = exp(1i*2*pi*t*f0); end
clear t



end