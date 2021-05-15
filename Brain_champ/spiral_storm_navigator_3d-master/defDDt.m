function [D,Dt] = defDDt
        % defines finite difference operator D
        % and its transpose operator
        
        D = @(U) grad(U);
        Dt = @(V) div(V);
        
        function DU = grad(U)
            % Forward finite difference operator 
            %(with circular boundary conditions)
            DU(:,:,:,1) = circshift(U,[1,0,0])- U;
            DU(:,:,:,2) = circshift(U,[0,1,0])- U;
            DU(:,:,:,3) = circshift(U,[0,0,1])- U;
        end
        
        function DtXY = div(V)
            DtXY = V(:,:,:,1)-circshift(V(:,:,:,1),[-1,0,0]);
            DtXY = DtXY + V(:,:,:,2) - circshift(V(:,:,:,2),[0,-1,0]);
            DtXY = DtXY + V(:,:,:,3) - circshift(V(:,:,:,3),[0,0,-1]);
        end
end
