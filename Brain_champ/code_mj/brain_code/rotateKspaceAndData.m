function [k,phase] = rotateKspaceAndData(k,params,linearflag)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

if(nargin<3)
    linearflag = false;
end

angles = params(1:3);
trans = [params(5),params(4),params(6)];
M = eul2rotm(pi/180*angles(:));
M = [0,1,0;1,0,0;0,0,1]'*M*[0,1,0;1,0,0;0,0,1];

sz = size(k);
k = reshape(k,prod(sz(1:3)),3);

k = k*M;

if(linearflag)
    nk = size(k,1);
    lin = [1:nk]/nk;
    k1 = bsxfun(@times,k,lin');
    phase = exp(1i*2*pi*k1*trans(:))';
else
    phase = exp(1i*2*pi*k*trans(:))';
end

k = reshape(k,sz);
phase = reshape(phase,sz(1:3));



end

