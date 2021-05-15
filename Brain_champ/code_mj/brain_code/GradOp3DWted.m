classdef GradOp3DWted
    %UNTITLED Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        eigDtD
        beta
        N
        wt
    end
    
    methods
        function obj = GradOp3DWted(N,beta,wt)
            obj.beta = beta;
            obj.N=N;
            if(nargin==3)
                obj.wt = wt;
            else
                obj.wt = ones(N(1),N(2),N(3));
            end
        end  
        
         function out = mtimes(obj,z) 
            z = reshape(z,obj.N);
            out = 0*z;
            if(obj.beta(1)>0) 
                D = z - circshift(z,[1,0,0,0]); D(1,:,:,:)=0;
                D = bsxfun(@times,D,obj.wt);
                out = out + obj.beta(1)*(D - circshift(D,[-1,0,0,0]));
            end
            
            if(obj.beta(2)>0) 
                D = z - circshift(z,[0,1,0,0]); D(:,1,:,:)=0;
                D = bsxfun(@times,D,obj.wt);
                out = out + obj.beta(2)*(D - circshift(D,[0,-1,0,0]));
            end
            
            if(obj.beta(3)>0) 
                D = z - circshift(z,[0,0,1,0]); D(:,:,1,:)=0;
                D = bsxfun(@times,D,obj.wt);
                out = out + obj.beta(3)*(D - circshift(D,[0,0,-1,0]));
            end
            
             if(obj.beta(4)>0) 
                D = z - circshift(z,[0,0,0,1]); D(:,:,:,1)=0;
                D = bsxfun(@times,D,obj.wt);
                out = out + obj.beta(3)*(D - circshift(D,[0,0,0,-1]));
            end      
            out = out(:);
         end
         
    
    function out = gradMag(obj,z) 
            z = reshape(z,obj.N);
            out = 0*z;
            D = z - circshift(z,[1,0,0,0]); D(1,:,:,:)=0;
            out = out + abs(D).^2;
            D = z - circshift(z,[0,1,0,0]); D(:,1,:,:)=0;
            out = out + abs(D).^2;
            D = z - circshift(z,[0,1,0,0]); D(:,:,1,:)=0;
            out = out + abs(D).^2;  
         end
    end

end



