FT= NUFFT(ktraj/N,1,0,0,[512,512]);

q = FT'*ones(size(ktraj));

