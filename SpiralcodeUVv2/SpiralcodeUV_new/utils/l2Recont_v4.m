function data = l2Recont_v4(kdata,FT,csm,lambda,N)
%function data = l2Recont(kdata,FT,csm,lambda,N)
%
% Arguments:
%   kdata   [np nv nf nc]   complex
%   FT      Fourier operator
%   csm     coil sens map
%   lambda      threshold
%   N               reconstucted image size
%
% Optional Arguments:

%
% Outputs:
%   data     [N N N Nt]      complex
%
% Ahmed, Abdul Haseeb <abdul-ahmed@uiowa.edu>

[~,~,nFrames,nCh] = size(kdata); 
Atb = zeros(N,N,nFrames);


for ii=1:nCh
    Atb = Atb + bsxfun(@times,FT'*kdata(:,:,:,ii),conj(csm(:,:,ii)));
end
lambda=max(abs(Atb(:)))*lambda;
 %Reg2= @(x) weighted_WT(abs(x),N1,nf,WT);
    %Reg2= @(x) sth((x),N1,nf,WT,0.01);

%     Reg1 = @(x) reshape(reshape(x,[N*N,nFrames])*(0.1*lambda*L),[N*N*nFrames,1]);
%     Reg2= @(x) weighted_TV(x,N1,nf);
%     Reg3= @(x) l1_norm_WT(x,N1,nf,WT);%%slight better
%     Reg3= @(x) l1_norm_TV((x),N,nFrames);



ATA = @(x) SenseATA(x,FT,csm,N,nFrames,nCh) + lambda*x;%
data = pcg(ATA,Atb(:),1e-5,50);
data = reshape(data,[N,N,nFrames]);

end

