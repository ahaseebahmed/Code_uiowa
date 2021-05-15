function out = GradOp(z,N,beta)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
 
            z = reshape(z,N);
            out = 0*z;
            if(beta(1)>0) 
                D = z - circshift(z,[1,0,0,0]); D(1,:,:,:)=0;
                out = out + beta(1)*(D - circshift(D,[-1,0,0,0]));
            end
            
            if(beta(2)>0) 
                D = z - circshift(z,[0,1,0,0]); D(:,1,:,:)=0;
                out = out + beta(2)*(D - circshift(D,[0,-1,0,0]));
            end
            
            if(beta(3)>0) 
                D = z - circshift(z,[0,0,1,0]); D(:,:,1,:)=0;
                out = out + beta(3)*(D - circshift(D,[0,0,-1,0]));
            end
            
            if(beta(4)>0) 
                D = z - circshift(z,[0,0,0,1]); D(:,:,:,1)=0;
                out = out + beta(4)*(D - circshift(D,[0,0,0,-1]));
            end        
            out = out(:);
         end

