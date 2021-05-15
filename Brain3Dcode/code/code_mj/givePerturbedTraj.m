function p = givePerturbedTraj(dphi,dtheta,p)

nks = size(p.ks,2);
ks = p.ks*p.factor*p.mtx_acq/p.mtx_reco; 

if(p.mtx_reco>p.mtx_acq)
    nS = nks;
else
    nS = floor(nks*p.mtx_reco/p.mtx_acq);
end
ks = ks(:,1:nS,:);

k = giveTraj(ks, p.phi, p.theta);

nCh = size(p.dd,1);
nangles = length(p.phi);

sinphi = sind(dphi);
cosphi = cosd(dphi);
sintheta = sind(-dtheta);
costheta = cosd(-dtheta);
  
   
M = [(costheta.*cosphi),(sinphi),-(sintheta.*cosphi)];
M = [M;-(costheta.*sinphi),(cosphi),(sintheta.*sinphi)];
M = [M;(sintheta),0,(costheta)];
k1_new = k*M';

p.dd = reshape(p.dd,nCh,nks,nangles);
p.dd = reshape(p.dd(:,1:nS,:),[nCh,nS*nangles]);
p.dcf = reshape(p.dcf(1:nS,:),nS*nangles,1);

osf = 2; % oversampling: 1.5 1.25
wg = 3; % kernel width: 5 7
sw = 8; % parallel sectors' width: 12 16

p.FT = gpuNUFFT(transpose(k1_new),ones(size(p.dcf)),osf,wg,sw,[1 1 1]*p.mtx_reco,[],true);
p.nS = nS;

end

