function M = giveRotMtx(dphi,dtheta)

dphi = -dphi;
sinphi = sind(dphi);
cosphi = cosd(dphi);
sintheta = sind(-dtheta);
costheta = cosd(-dtheta);
  
   
M = [(cosphi),-(costheta.*sinphi),(sintheta.*sinphi)];
M = [M;(sinphi),(costheta.*cosphi),-(sintheta.*cosphi)];
M = [M;0,(sintheta),(costheta)];

M = [M,[0;0;0]];
M = [M;0,0,0,1];