function out = giveATA(x,FT,dcf,N)


out = FT*reshape(x,N);
out = bsxfun(@times,out,col(dcf));
out = FT'*out;
out = out(:);

end

