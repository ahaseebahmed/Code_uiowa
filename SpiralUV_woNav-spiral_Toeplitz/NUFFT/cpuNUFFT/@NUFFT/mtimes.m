function ress = mtimes(a,bb)

 if a.adjoint
     % Multicoil non-Cartesian k-space to Cartesian image domain
     % nufft for each coil and time point
     for ch=1:size(bb,2)
         b = bb(:,ch).*a.w(:);
         res(:,:,ch) = reshape(nufft_adj(b(:),a.st)/sqrt(prod(a.imSize)),a.imSize(1),a.imSize(2));
     end
     
     % compensate for undersampling factor
     %res=res*size(a.b1,1)*pi/2/size(a.w,2);     
     % coil combination for each time point
    ress=sum(res.*conj(a.b1),3)./sum(abs((a.b1)).^2,3);
  
 else
     % Cartesian image to multicoil non-Cartesian k-space 
     for ch=1:size(a.b1,3)
        res=bb.*a.b1(:,:,ch);
        ress(:,ch) = nufft(res,a.st)/sqrt(prod(a.imSize)).*a.w(:);
     end

 end

