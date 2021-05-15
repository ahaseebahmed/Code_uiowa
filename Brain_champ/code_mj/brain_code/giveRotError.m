function [error,combined] = giveRotError(angles,p1,p2)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

dphi = angles(1);
dtheta = angles(2);
recon1 = p1.FT'*bsxfun(@times,p1.dd',p1.dcf);
recon2 = p2.FT'*bsxfun(@times,p2.dd',p2.dcf);

combined = recon1 + rotImage(recon2,-dphi,-dtheta);
combined2 = rotImage(combined,dphi,dtheta);

e1 = double(p1.FT*combined - p1.dd');
e1 = bsxfun(@times,e1,double(p1.dcf).^2);
error1 = sum(abs(e1(:)).^2);

e2 = double(p2.FT*combined - p2.dd');
e2 = bsxfun(@times,e2,double(p2.dcf).^2);
error2 = sum(abs(e2(:)).^2);
error = error1+error2;

end

