[fname,pathname] = uigetfile('*.mat');
filename=[pathname fname];

fprintf("Reading from %s\n",filename);

load(filename);

recon = reshape((U1)*D',[paramOut.n,paramOut.n,paramOut.nf]);
recon = fftshift(fftshift(recon,1),2);
figure(2)
for k=1:paramOut.nf;imagesc((abs((recon(:,:,k)))),[0,2e-4]); colormap gray;pause(0.05);end
