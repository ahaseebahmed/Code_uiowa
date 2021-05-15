function p = transformKspaceAndDataCPU(tform,p,factor)

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
trans = factor*[tform.T(4,2),tform.T(4,1),tform.T(4,3)];
phase = exp(1i*2*pi*p.k*trans');

p.dd = reshape(p.dd,nCh,p.nS,nangles);
p.dd = reshape(p.dd(:,p.indices,:),[nCh,nks*nangles]);
p.dd = p.dd.*(phase');
p.dcf = reshape(p.dcf(p.indices,:),nks*nangles,1);


end

