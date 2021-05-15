function [A,B]=creatA(kloc,N)
kx=[];
ky=[];
kx=real(kloc);
ky=imag(kloc);
n=-N/2:1:N/2-1;
%n=1:1:N;
%kx=gpuArray(kx);ky=gpuArray(ky);n=gpuArray(n) ; N=gpuArray(N);
A=exp(-1i*2*pi*(kx')*n/N)*1/N;
B=exp(-1i*2*pi*(ky')*n/N)*1/N;
end

