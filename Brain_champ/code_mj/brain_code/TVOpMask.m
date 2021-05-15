classdef TVOpMask
    %UNTITLED Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        eigDtD
        beta
        N
    end
    
    methods
        function obj = TVOp(N,beta,useGPU)
            obj.beta = beta;
            obj.N=N;
            obj.eigDtD = abs(beta(1)*fftn([1 -1], N)).^2 + abs(beta(2)*fftn([1 -1]', N)).^2 ;
            d_tmp = zeros(2,2,2);
            d_tmp(1,1,1)= 1; d_tmp(1,1,2)= -1;
            obj.eigDtD   = obj.eigDtD + abs(beta(3)*fftn(d_tmp, N)).^2;
            if(useGPU)
                obj.eigDtD = gpuArray(obj.eigDtD);
            end
        end  
        
         function z1 = mtimes(obj,z)
            z = reshape(z,obj.N);
            z1 = (z);
            z1 = ifftn(fftn(z1).*obj.eigDtD);
            z1 = (z1); 
            z1 = z1(:);
         end
         
         function [Dux,Duy,Duz] = ForwardD(obj,U)
            frames = size(U, 3);
            Dux = obj.beta(1)*[diff(U,1,2), U(:,1,:) - U(:,end,:)];
            Duy = obj.beta(2)*[diff(U,1,1); U(1,:,:) - U(end,:,:)];
            Duz(:,:,1:frames-1) = obj.beta(3)*diff(U,1,3); 
            Duz(:,:,frames)     = obj.beta(3)*(U(:,:,1) - U(:,:,end));
         end
        
         function DtXYZ = Dive(obj,X,Y,Z)
            frames = size(X, 3);
            DtXYZ = [X(:,end,:) - X(:, 1,:), -diff(X,1,2)];
            DtXYZ = obj.beta(1)*DtXYZ + obj.beta(2)*[Y(end,:,:) - Y(1, :,:); -diff(Y,1,1)];
            Tmp(:,:,1) = Z(:,:,end) - Z(:,:,1);
            Tmp(:,:,2:frames) = -diff(Z,1,3);
            DtXYZ = DtXYZ + obj.beta(3)*Tmp;
         end

         function [u1,u2,u3] = shrink(v1,v2,v3)
             v  = sqrt(abs(v1).^2 + abs(v2).^2 + abs(v3).^2);
             v(v==0) = 1;
             v  = max(v - 1/rho, 0)./v;
             u1 = v1.*v;
             u2 = v2.*v;
             u3 = v3.*v;
         end

    end
end



