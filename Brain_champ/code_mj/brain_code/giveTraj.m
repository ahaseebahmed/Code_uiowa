function k = giveTraj(ks, phi, theta)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

  ks = squeeze(ks);
  nks = length(ks);
  nviews = size(phi,2);
  k = zeros(nks,nviews,3,class(ks));
            
  sinphi = sind(phi);
  cosphi = cosd(phi);
  sintheta = sind(theta);
  costheta = cosd(theta);
            
  k(:,:,1) =  ks(:,1)*(costheta.*cosphi) + ...
                ks(:,2)*(sinphi) - ...
                ks(:,3)*(sintheta.*cosphi);
  k(:,:,2) =  -ks(:,1)*(costheta.*sinphi) + ...
                ks(:,2)*(cosphi) + ...
                ks(:,3)*(sintheta.*sinphi);
  k(:,:,3) = ks(:,1)*(sintheta) + ...
                ks(:,3)*(costheta);
  k = reshape(k,nks*nviews,3);


end

