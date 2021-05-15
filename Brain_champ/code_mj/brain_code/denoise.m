function pdenoised = denoise(p)

disp('Starting denoising');
pdenoised = gather(p);
pdenoised = abs(pdenoised);mx = max(abs(pdenoised(:)));
pdenoised = pdenoised./mx;
    
[hfinal, ho, SNRo, hbg, SNRbg] = MRINoiseEstimation(pdenoised, 0, 0);
%MRIdenoised = MRIDenoisingONLM(pdenoised, hfinal, 2, 1,  3, 1, 0);
%MRIdenoised = MRIDenoisingODCT(pdenoised, hfinal, 1, 1, 0);
 MRIdenoised = MRIDenoisingPRINLM(pdenoised, hfinal, 2, 1, 0);
pdenoised = MRIdenoised*mx.*exp(1i*angle(p));

disp('Done denoising');


end

