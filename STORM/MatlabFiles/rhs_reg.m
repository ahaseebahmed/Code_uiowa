
function [reg_term,wt] = rhs_reg(U,n,r,LamMtx,threshold,ww)

        
    wt = abs(reshape(U,n^2,r))-threshold;
    
    wt = wt > 0;
    %wt = wt.*(wt > 0);
    %wt = wt./(abs(reshape(U,n^2,r))+0.01*threshold);
    
    reg_term = reshape(U,n^2,r).*wt;
    
    if(nargin>5)
        reg_term(:,1:end-1) = reg_term(:,1:end-1).*reshape(ww,[n^2 r-1]);
    end
     reg_term = reshape(reg_term,n^2,r)*LamMtx;  
     reg_term = reg_term(:); 
end

