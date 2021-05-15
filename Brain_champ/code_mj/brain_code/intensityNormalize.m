clear all

fname = '/Users/jcb/abdul_brain/reconData/denoised_IR_cg_600_auto';
load([fname,'.mat'])
addpath(genpath('/Users/jcb/OneDrive - University of Iowa/WORK/Programs/Medical Image Reader and Viewer'));

%denoised(isnan(denoised))=0;
%noisy(isnan(noisy))=0;
denoised1 = abs(p);

threshold = 0.1;
%lambda = 2.5*size(denoised,1)/100;    % Higher smoothness for higher matrix size
lambda = 100;

%denoised1 = denoised(50:end-50,50:end-50,150:end-150);
[denoised_norm,biasfield] = giveNormalizedImage(denoised,threshold,lambda,true);

noisy_norm = noisy(50:end-50,50:end-50,150:end-150).*biasfield;
save([fname,'_normalized.mat'],'denoised_norm');