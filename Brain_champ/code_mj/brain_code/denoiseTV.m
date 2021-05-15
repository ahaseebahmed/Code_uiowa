function pdenoised = denoiseTV(p)

disp('Starting denoising');
pdenoised = gather(p);
mx = max(abs(pdenoised(:)));
pdenoised = pdenoised./mx;

opts.beta    = [1 1 1];
opts.print   = false;
opts.method  = 'l2';
opts.max_itr = 50;
opts.tol = 1e-5;

[hfinal, ho, SNRo, hbg, SNRbg] = MRINoiseEstimation(abs(pdenoised), 0, 0);
out = deconvtvl2(gpuArray(pdenoised), 1/hfinal, opts);

pdenoised = gather(out.f.*mx);
disp('Denoising done');

