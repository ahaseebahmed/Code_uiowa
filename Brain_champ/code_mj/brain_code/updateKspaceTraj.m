function p = updateKspaceTraj(p,k,d)


p.k = k;
osf = 2; % oversampling: 1.5 1.25
wg = 3; % kernel width: 5 7
sw = 8; % parallel sectors' width: 12 16

p.k=k;
if(nargin == 3)
    p.d = d;
end
p.FT = gpuNUFFT(transpose(p.k),ones(size(p.dcf)),osf,wg,sw,[1 1 1]*p.mtx_reco,[],true);

end

