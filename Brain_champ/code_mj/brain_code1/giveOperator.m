function p = giveOperator(tform,phi,theta,ks,dcf,mtx_acq,mtx_reco,factor,dd)

[nsamples,nangles] = size(dcf); 
p.mtx_reco=mtx_reco;
p.phi = phi(:)';
p.theta =  theta(:)';

p.nS = size(ks,2);
p.ks = ks*factor*mtx_acq/p.mtx_reco; 
p.indices = squeeze(find(sum(abs(p.ks).^2,3)<0.25));
p.ks = p.ks(:,p.indices,:);
p.dcf = reshape(dcf,[nsamples,nangles]);
p.nks = length(p.ks);

if(nargin<9)
    dd = zeros(1,nangles*nsamples);
end
p.dd = dd;
p.nangles = nangles;
p = transformKspaceAndData(tform,p,factor);

end

