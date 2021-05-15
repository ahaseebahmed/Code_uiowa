function out = SenseATA(x,Q,csm,N,nFrames,nCh)

  x = reshape(x,[N,N,nFrames]);
  ind = N/2:N/2+N-1;
  
  out = (zeros(N,N,nFrames));
  z= (zeros(2*N,2*N,nCh));
  for i=1:nFrames
        xnew = z;
        xnew(ind,ind,:) = bsxfun(@times,squeeze(x(:,:,i)),csm);
        xnew = ifft2(bsxfun(@times,Q(:,:,i),fft2(xnew)));
        out(:,:,i) = sum(xnew(ind,ind,:).*conj(csm),3);
      end
  out = out(:);
    
end