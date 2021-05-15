function p = updateKspaceAndData(tform,p,factor)

if(nargin<3)
    factor = 1;
end

nks = length(p.ks);
p.k = giveTraj(p.ks, p.phi, p.theta);
nCh = size(p.dd,1);
nangles = length(p.phi);


M = tform.T(1:3,1:3);
M = [0,1,0;1,0,0;0,0,1]'*M*[0,1,0;1,0,0;0,0,1];

p.k = p.k*M';

%trans = factor*[tform.T(4,2),tform.T(4,1),tform.T(4,3)];
%phase = exp(1i*2*pi*p.k*trans');

%p.dd = p.dd.*(phase');

% p.dd = reshape(p.dd,nCh,p.nS,nangles);
% p.dd = reshape(p.dd(:,1:nks,:),[nCh,nks*nangles]);
% p.dcf = reshape(p.dcf(1:nks,:),nks*nangles,1);

osf = 2; % oversampling: 1.5 1.25
wg = 3; % kernel width: 5 7
sw = 8; % parallel sectors' width: 12 16

p.FT = gpuNUFFT(transpose(p.k),ones(size(p.dcf)),osf,wg,sw,[1 1 1]*p.mtx_reco,[],true);

end

