function csm = csm(coilimages,szFilter,szBlk)


sos = sqrt(sum(abs(coilimages).^2,4));
SOS = fftshift(fftn(fftshift(sos)));

[nx,ny,nz,nCh] = size(coilimages);
data = zeros(nx,ny,nz,nCh);
for i=1:nCh
    data(:,:,:,i) = fftshift(fftn(fftshift(coilimages(:,:,:,i))));
end

cx = nx/2; cy=ny/2; cz = nz/2;


Neqns = (2*szBlk+1).^3;
Ncoeffs = (2*szFilter+1).^3;
A = zeros(Ncoeffs,Neqns);
b=zeros(Ncoeffs,Nch);

n=1;
for i=cx-szBlk:cx+szBlk,
    for j=cy-szBlk:cy+szBlk,
        for k=cz-szBlk:cz+szBlk,
            temp = SOS(i+szFilter:-1:i-szFilter,j+szFilter:-1:j-szFilter,k+szFilter:-1:k-szFilter);
            A(:,n) = temp(:);
            b(n,:) = squeeze(data(i,j,k,:));
            n=n+1;
        end
    end
end

coeffs = transpose(A)\b;
coeffs = reshape(coeffs,2*szFilter+1,2*szFilter+1,2*szFilter+1,nCh);

csm = zeros(nx,ny,nz,nCh);
csm(cx-szFilter:cx+szFilter,cx-szFilter:cx+szFilter,cx-szFilter:cx+szFilter,:) = coeffs;

% n=1;
% for i=cx+szFilter:-1:cx-szFilter,
%     for j=cy+szFilter:-1:cy-szFilter,
%         for k=cz+szFilter:-1:cz-szFilter,
%             csm(i,j,k) = coeffs(n);
%             n=n+1;
%         end
%     end
% end
% 
for i=1:nCh
    csm(:,:,:,i) = ifftshift(ifftn(ifftshift(csm(:,:,:,i))));
end% 
% 
