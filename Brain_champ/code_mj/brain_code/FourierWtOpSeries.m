classdef FourierWtOpSeries
    %UNTITLED Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        kspaceWeights
        Sense
        Nc
        Nt
        N
    end
    
    methods
        function obj = FourierWtOpSeries(kspaceWeights,sense,useGPU)
            obj.kspaceWeights = kspaceWeights;
            obj.Sense = sense;
            obj.Nc = size(sense,4);
            obj.N = size(sense,1);
            obj.Nt = size(kspaceWeights,4);
        end  
        
         function out = mtimes(obj,z)
              z = reshape(z,obj.N,obj.N,obj.N,obj.Nt);
              out = zeros(obj.N,obj.N,obj.N,obj.Nt,class(z));
                for k=1:obj.Nt,  
                    wts = gpuArray(obj.kspaceWeights(:,:,:,k));
                    for i=1:obj.Nc
                        z1 = z(:,:,:,k).*(obj.Sense(:,:,:,i));
                        z2 = zeros(2*obj.N*[1,1,1],class(z));
                        z2(1:obj.N,1:obj.N,1:obj.N)=z1;
                        z2 = ifftshift(ifftn(fftn(fftshift(z2)).*obj.kspaceWeights(:,:,:,k)));
                        %z2 = ifftn(fftn(z1,2*obj.N*[1,1,1]).*obj.kspaceWeights(:,:,:,k));
                        z1 = (z2(1:obj.N,1:obj.N,1:obj.N)); 
                        out(:,:,:,k) = out(:,:,:,k) +z1.*conj(obj.Sense(:,:,:,i));
                    end
            end
            clear s w;
            out = out(:);
         end
        
         function delete(obj)
             clear obj.Sense;
             clear obj.kspaceWeights;
         end

    end
end



