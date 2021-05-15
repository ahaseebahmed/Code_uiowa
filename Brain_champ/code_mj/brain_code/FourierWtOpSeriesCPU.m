classdef FourierWtOpSeriesCPU
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
        function obj = FourierWtOpSeriesCPU(kspaceWeights,sense)
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
                    for i=1:obj.Nc
                        z1 = z(:,:,:,k).*(obj.Sense(:,:,:,i));
                        z1 = fftn(fftshift(z1));
                        z1 = z1.*obj.kspaceWeights(:,:,:,k);
                        z1 = ifftshift(ifftn(z1));
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



