function [ HFEN_m ] = hfen( U,r )
%HFEN Summary of this function goes here
%Detailed explanation goes here
%r- ideal image time series
%U - reconstructed time series
for t = 1:size(r,3)
    alpha = sum(dot(U(:,:,t),r(:,:,t)))/(sum(dot(U(:,:,t),U(:,:,t))));
U(:,:,t)=(alpha)*U(:,:,t);
p=15; % size of 15*15 pixels
d=1.5; %standard deviation of 1.5 pixels.
HFEN_m(t)=norm(imfilter(abs(U(:,:,t)),fspecial('log',p,d)) - imfilter(abs(r(:,:,t)),fspecial('log',p,d)),'fro')./norm( imfilter(abs(r(:,:,t)),fspecial('log',p,d)),'fro');
end
HFEN_m=mean(HFEN_m);