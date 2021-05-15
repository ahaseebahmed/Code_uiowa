classdef FourierWtOp
    %UNTITLED Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        kspaceWeights
        Sense
        Nc
        N
    end
    
    methods
        function obj = FourierWtOp(kspaceWeights,sense,useGPU)
            obj.kspaceWeights = kspaceWeights;
            obj.Sense = sense;
            obj.Nc = size(sense,4);
            obj.N = size(sense,1);
            if(nargin==3 && useGPU)
                obj.kspaceWeights = gpuArray(kspaceWeights);
                obj.Sense = gpuArray(sense);
            end
        end  
        
         function out = mtimes(obj,z)
            z = reshape(z,obj.N,obj.N,obj.N);
              out = gpuArray(zeros(obj.N,obj.N,obj.N));
            for i=1:obj.Nc
                z1 = z.*obj.Sense(:,:,:,i);
                z1 = fftshift(z1);
                z1 = ifftn(fftn(z1).*obj.kspaceWeights);
                z1 = ifftshift(z1); 
                out = out +z1.*conj(obj.Sense(:,:,:,i));
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



