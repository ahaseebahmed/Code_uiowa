function [coilimages] = coil_sens_map(Atb,Q,useGPU)

[N,~,~,nCh] = size(Atb); 
coilimages=zeros(N,N,nCh);

Atb = squeeze(sum(Atb,3));
Q = squeeze(sum(Q,3));

if(useGPU)
    ATA = @(x) SenseATA_GPU(x,gpuArray(Q),gpuArray(ones(N)),N,1,1)+0.01*x;
    Atb = gpuArray(Atb);
else
    ATA = @(x) SenseATA(x,(Q),ones(N),N,1,1)+0.1*x;
end

for i=1:nCh
    temp = Atb(:,:,i);
    coilimages(:,:,i)= gather(reshape(pcg(ATA,temp(:),1e-4,100),[N,N]));
end

end