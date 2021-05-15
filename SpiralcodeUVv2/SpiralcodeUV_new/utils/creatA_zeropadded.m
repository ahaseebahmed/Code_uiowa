function [A,B]=creatA_zeropadded(kloc,N,K)

kx=real(kloc);
ky=imag(kloc);
n=-K/2:1:K/2-1;

A=exp(-1i*2*pi/N*(kx')*n)*1/(N);
B=exp(-1i*2*pi/N*(ky')*n)*1/(N);

