function out = XL(x,N,Nframes,L)
   out = reshape(x,[N*N,Nframes])*L;
   out = out(:);
end