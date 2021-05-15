function out = rotImage(in,dphi,dtheta)  

% phi (rotation in x y plane; orthogonal to z)
% theta (rotation in y z plane, orthogonal to x)

  sinphi = sind(dphi);
  cosphi = cosd(dphi);
  sintheta = sind(dtheta);
  costheta = cosd(dtheta);
  
  M = [(costheta.*cosphi),(sinphi),(sintheta.*cosphi)];
  M = [M;-(costheta.*sinphi),(cosphi),(sintheta.*sinphi)];
  M = [M;(sintheta),0,(costheta)];
  
  eulZYX = 180/pi*rotm2eul(M);
   
  out = imrotate3(real(in),-dphi,[0,0,1],'crop','cubic')+  1i*imrotate3(imag(in),-dphi,[0,0,1],'crop','cubic');
  out = imrotate3(real(out),dtheta,[1,0,0],'crop','cubic')+  1i*imrotate3(imag(out),-dtheta,[0,1,0],'crop','cubic');
 % out = imrotate3(real(out),eulZYX(3),[1,0,0],'crop','cubic')+  1i*imrotate3(imag(out),eulZYX(3),[1,0,0],'crop','cubic');
