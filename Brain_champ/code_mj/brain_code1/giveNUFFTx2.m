function out = giveNUFFTx2(x,FT,mtx_reco,k)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

ind = zeros(length(x),8);
ind(:,1) = x.*exp(1i*pi*k'*[mtx_reco,mtx_reco,mtx_reco]');
ind(:,2) = x.*exp(1i*pi*k'*[mtx_reco,-mtx_reco,mtx_reco]');
ind(:,3) = x.*exp(1i*pi*k'*[-mtx_reco,mtx_reco,mtx_reco]');
ind(:,4) = x.*exp(1i*pi*k'*[-mtx_reco,-mtx_reco,mtx_reco]');
ind(:,5) = x.*exp(1i*pi*k'*[mtx_reco,mtx_reco,-mtx_reco]');
ind(:,6) = x.*exp(1i*pi*k'*[mtx_reco,-mtx_reco,-mtx_reco]');
ind(:,7) = x.*exp(1i*pi*k'*[-mtx_reco,mtx_reco,-mtx_reco]');
ind(:,8) = x.*exp(1i*pi*k'*[-mtx_reco,-mtx_reco,-mtx_reco]');

q = (FT'*ind);

out = zeros(2*mtx_reco*[1,1,1]);
out(1:mtx_reco,1:mtx_reco,1:mtx_reco) = q(:,:,:,1);
out(1:mtx_reco,mtx_reco+1:2*mtx_reco,1:mtx_reco) = q(:,:,:,2);
out(mtx_reco+1:2*mtx_reco,1:mtx_reco,1:mtx_reco) = q(:,:,:,3);
out(mtx_reco+1:2*mtx_reco,mtx_reco+1:2*mtx_reco,1:mtx_reco) = q(:,:,:,4);

out(1:mtx_reco,1:mtx_reco,mtx_reco+1:2*mtx_reco) = q(:,:,:,5);
out(1:mtx_reco,mtx_reco+1:2*mtx_reco,mtx_reco+1:2*mtx_reco) = q(:,:,:,6);
out(mtx_reco+1:2*mtx_reco,1:mtx_reco,mtx_reco+1:2*mtx_reco) = q(:,:,:,7);
out(mtx_reco+1:2*mtx_reco,mtx_reco+1:2*mtx_reco,mtx_reco+1:2*mtx_reco) = q(:,:,:,8);
out = fftshift(out);


end

