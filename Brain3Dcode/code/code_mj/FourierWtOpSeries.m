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
             %if(nargin==3 && useGPU)
                obj.kspaceWeights = (kspaceWeights);
                obj.Sense = (sense);
             %end
        end  
        
         function out = mtimes(obj,z)
              z = reshape(z,obj.N,obj.N,obj.N,obj.Nt);
              out = (zeros(obj.N,obj.N,obj.N,obj.Nt));
                for k=1:obj.Nt,  
                    wts = (obj.kspaceWeights(:,:,:,k));
                    for i=1:obj.Nc
                        z1 = z(:,:,:,k).*(obj.Sense(:,:,:,i));
                        z2 = (zeros(2*obj.N*[1,1,1]));
                        z2(1:obj.N,1:obj.N,1:obj.N)=z1;clear z1;
                        z2 = ifftn(fftn(z2).*wts);
                        %z2 = ifftn(fftn(z1,2*obj.N*[1,1,1]).*obj.kspaceWeights(:,:,:,k));
                        z1 = (z2(1:obj.N,1:obj.N,1:obj.N)); clear z2;
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



