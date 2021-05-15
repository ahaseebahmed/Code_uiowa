classdef ToeplitzOp
    %UNTITLED Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        kspaceWeights
        Sense
        Nc
        N
        holder
    end
    
    methods
        function obj = ToeplitzOp(kspaceWeights,sense,useGPU)
            obj.kspaceWeights = kspaceWeights;
            obj.Sense = sense;
            obj.Nc = size(sense,4);
            obj.N = size(sense,1);
            obj.holder = complex(zeros(2*obj.N*[1,1,1])); 
            if(nargin==3 && useGPU)
                obj.kspaceWeights = gpuArray(kspaceWeights);
                obj.Sense = gpuArray(sense);
                obj.holder = gpuArray(obj.holder);
            end
        end  
        
         function out = mtimes(obj,z)
            z = reshape(z,obj.N,obj.N,obj.N);
            out = zeros(obj.N,obj.N,obj.N);
            for i=1:obj.Nc
                z1 = z.*obj.Sense(:,:,:,i);
                test = obj.holder;
                test(1:obj.N,1:obj.N,1:obj.N,:) = z1;
                z1 = fftshift(test);
                z1 = ifftn(fftn(z1).*obj.kspaceWeights);
                z1 = ifftshift(z1); 
                z1 = z1(1:obj.N,1:obj.N,1:obj.N);
                out = out +z1.*conj(obj.Sense(:,:,:,i));
            end
            out = out(:);
         end
        
          

    end
end



