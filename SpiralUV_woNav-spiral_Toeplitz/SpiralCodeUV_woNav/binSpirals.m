function [kdata_com,ktraj_com] = binSpirals(kdata,ktraj,NsamplesToKeep,spiralsToBin,spiral)

if(mod(spiralsToBin,2)==0)
    display('spiralsToBin need to be odd')
    return;
end

[NSamplesPerSpiral,Nspirals,nFrames,nCoils] = size(kdata);
if(NSamplesPerSpiral < NsamplesToKeep)
    display('Too many spiral points; choose less')
    return;
end
NE=size(kdata,1);
if spiral
kdata = reshape(kdata(1:NsamplesToKeep,:,:,:),[NsamplesToKeep,Nspirals*nFrames,nCoils]);
ktraj = reshape(ktraj(1:NsamplesToKeep,:,:,:),[NsamplesToKeep,Nspirals*nFrames]);
else
kdata = reshape(kdata((NE/2)+1-(NsamplesToKeep/2):(NE/2)+(NsamplesToKeep/2),:,:,:),[NsamplesToKeep,Nspirals*nFrames,nCoils]);
ktraj = reshape(ktraj((NE/2)+1-(NsamplesToKeep/2):(NE/2)+(NsamplesToKeep/2),:,:,:),[NsamplesToKeep,Nspirals*nFrames]);
end
binHalf = floor(spiralsToBin/2);
kdata_com = zeros(NsamplesToKeep,spiralsToBin,nFrames,nCoils);
ktraj_com = zeros(NsamplesToKeep,spiralsToBin,nFrames);

matrix = [];
i=1;
for index = floor(Nspirals/2):Nspirals:Nspirals*nFrames
    currIndices = [index - binHalf: index+binHalf];
    if(currIndices(1)<1)
        currIndices = currIndices - currIndices(1)+1;
    end
    if(currIndices(end)>Nspirals*nFrames)
        currIndices = currIndices + (Nspirals*nFrames-currIndices(end));
    end
    
    kdata_com(:,:,i,:) = reshape(kdata(:,currIndices,:),[NsamplesToKeep,spiralsToBin,1,nCoils]);
    ktraj_com(:,:,i,:) = reshape(ktraj(:,currIndices),[NsamplesToKeep,spiralsToBin,1]);

    i=i+1;
end
