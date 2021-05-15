function [k,d] = shiftKspaceData(k,d,params)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

angles = params(1:3);
trans = params(4:6);
M = eul2rotm(pi/180*angles(:));
M = [0,1,0;1,0,0;0,0,1]'*M*[0,1,0;1,0,0;0,0,1];
k = k*M';

phase = exp(1i*2*pi*k*trans(:))';
d = bsxfun(@times,d,phase);


end

