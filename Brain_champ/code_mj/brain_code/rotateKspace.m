function k = rotateKspace(k,angles)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

M = eul2rotm(pi/180*angles(:));
M = [0,1,0;1,0,0;0,0,1]'*M*[0,1,0;1,0,0;0,0,1];

k = k*M';

end

