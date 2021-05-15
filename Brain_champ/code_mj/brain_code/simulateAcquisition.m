function [synthetic_data,p,out] = simulateAcquisition(im,parameterArray,ks,dcf,mtx_acq,tform)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

[mtx_reco,~,~,nCh] = size(im);

if(nargin<7)
    tform = affine3d(eye(4));
end

phi = parameterArray(:,1);
theta = parameterArray(:,2);

p = giveOperator(tform,phi,theta,ks,dcf,mtx_acq,mtx_reco,1.2);
synthetic_data = p.FT*im;
out = zeros(nCh,p.nS,p.nangles);
out(:,p.indices,:) = reshape(synthetic_data,[nCh,size(p.indices),p.nangles]);

end

