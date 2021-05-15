function [maskfinal,sos,csm] = giveCSMWithBias(coilimages,szFilter,szBlk)

disp('Computing the CSM');

sos = sqrt(sum(abs(coilimages).^2,4));

SOS = (fftn(fftshift(sos)));
[nx,ny,nz,nCh] = size(coilimages);

% finding the mask as the convex hull of a segmented volume
%----------------------------------------------------------
mx = max(sos(:));
mask = abs(sos)>0.1*mx;
mask1 = abs(sos)>0.12*mx;

SOSMASKED = fftshift(fftn(fftshift(sos.*mask1)));
MASK = fftshift(fftn(fftshift(mask1)));


Ns = ceil(nx/100);
    
mask = mask(1:Ns:end,1:Ns:end,1:Ns:end);
rgbind = find(mask);
[r,g,b] = ind2sub(size(mask),rgbind);
S = alphaShape(r,g,b,inf);
%plot(S)

[nx,ny,nz] = size(mask);
[rall,gall,ball] = ndgrid(1:nx,1:ny,1:nz);
rgball = [rall(:),gall(:),ball(:)];
inlist = inShape(S,rgball);
mask = reshape(inlist,[nx,ny,nz]);
[nxfinal,nyfinal,nzfinal] = size(sos);
[rall,gall,ball] = meshgrid(1:nx,1:ny,1:nz);
[rfinal,gfinal,bfinal] = meshgrid(1:nxfinal,1:nyfinal,1:nzfinal);
maskfinal = interp3(rall,gall,ball,double(mask),rfinal*nx/nxfinal,gfinal*nx/nxfinal,bfinal*nx/nxfinal);
maskfinal(isnan(maskfinal))=0;

%%

if(nargin==3)
    SOS = gather(fftshift(SOS));

    data = zeros(nxfinal,nyfinal,nzfinal,nCh);
    for i=1:nCh
        data(:,:,:,i) = gather(fftshift(fftn(fftshift(gpuArray(coilimages(:,:,:,i))))));
    end

    cx = nxfinal/2; cy=nyfinal/2; cz = nzfinal/2;

    Neqns = (2*szBlk+1).^3;
    Ncoeffs = (2*szFilter+1).^3;
    A = zeros(Ncoeffs,Neqns);
    A1 = zeros(Ncoeffs,Neqns);

    b=zeros(Ncoeffs,nCh);
    b1 = zeros(Ncoeffs,1);

    n=1;
    for i=cx-szBlk:cx+szBlk,
        for j=cy-szBlk:cy+szBlk,
            for k=cz-szBlk:cz+szBlk,
                temp = SOS(i+szFilter:-1:i-szFilter,j+szFilter:-1:j-szFilter,k+szFilter:-1:k-szFilter);
                A(:,n) = temp(:);
                temp = MASK(i+szFilter:-1:i-szFilter,j+szFilter:-1:j-szFilter,k+szFilter:-1:k-szFilter);
                A1(:,n) = temp(:);
                b(n,:) = squeeze(data(i,j,k,:));
                b1(n,:) = squeeze(SOSMASKED(i,j,k));
                n=n+1;
            end
        end
    end

    coeffs = transpose(A)\b;
    coeffs = reshape(coeffs,2*szFilter+1,2*szFilter+1,2*szFilter+1,nCh);

    csm = zeros(nxfinal,nyfinal,nzfinal,nCh);
    csm(cx-szFilter:cx+szFilter,cx-szFilter:cx+szFilter,cx-szFilter:cx+szFilter,:) = coeffs;


    coeffs = transpose(A1)\b1;
    coeffs = reshape(coeffs,2*szFilter+1,2*szFilter+1,2*szFilter+1);
    bias = zeros(nxfinal,nyfinal,nzfinal);
    bias(cx-szFilter:cx+szFilter,cx-szFilter:cx+szFilter,cx-szFilter:cx+szFilter,:) = coeffs;
    bias = gather(ifftshift(ifftn(ifftshift(gpuArray(bias)))));


    for i=1:nCh
        csm(:,:,:,i) = gather(ifftshift(ifftn(ifftshift(gpuArray(csm(:,:,:,i))))));
        csm(:,:,:,i) = csm(:,:,:,i).*bias;
    end
end

disp('CSM done');

%sos = sqrt(sum(abs(csm).^2,4));
%cmap_norm = bsxfun(@times,csm,1./sos);