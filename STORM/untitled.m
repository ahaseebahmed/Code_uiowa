clear all;
%addpath('gspbox');
%gsp_start;
addpath('MatlabFiles');

load bCom;
%%
q = fft(bCom_img,[],2);
q(:,param.nf/2-param.nf/4-10:param.nf/2+param.nf/4+10)=0;
bCom_smoothed = ifft(q,[],2);

[~, bCom_est, L] = estimateLapKernelLR(bCom_smoothed, 0.25, 0.1,0);
[~,SBasis,D] = svd(L);
D= D(:,end-param.Nbasis:end);
SBasis = SBasis(end-param.Nbasis:end,end-param.Nbasis:end);
%%
load temp;

c1 = AhbVh(b,S,D',conj(csm),param.n,size(D,2),param.nf,nvchannels,param.useGPU);
    LamMtx = SBasis*param.lambdaCoeffs;
    gradU = @(z)(AhAUVVh_reg(z,D',S,csm,param.n,size(D,2),param.nf,nvchannels,LamMtx,[],[],param.useGPU,gwin));
    U = [];
    tic;[U,~,~,~,err] = pcg(gradU,c1,10^-10,param.CoeffsCGIterations);toc;
    Ul2 = gather(reshape(U,[param.n^2,size(D,2)]));
%%
[reg_term,wt] = rhs_reg(U,param.n,param.Nbasis+1,LamMtx, param.eps,gwin);

 LamMtx = SBasis*2*param.lambdaCoeffs; 
 gradU = @(z)(AhAUVVh_reg(z,D',S,csm,param.n,size(D,2),param.nf,nvchannels,LamMtx,[],[],param.useGPU,gwin));
 tic;[U,~,~,~,err] = pcg(gradU,c1 + reg_term,10^-10,15,[],[],U);toc;
 Ul1_new = gather(reshape(U,[param.n^2,size(D,2)]));
 %%
 for i=100:350 
     q1 = U1(:,1:end)*D(i,1:end)'; 
     q1= (fftshift(abs(reshape(q1,512,512))));
     q2 = Ul2(:,1:end)*D(i,1:end)'; 
     q2= (fftshift(abs(reshape(q2,512,512)))); 
     q = [q1(150:350,150:350),q2(150:350,150:350)];
     imagesc(q,[0,6e-4]);
     title(num2str(i));
     pause(0.01);
 end
  