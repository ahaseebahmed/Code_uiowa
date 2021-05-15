classdef SenseOp
    %UNTITLED Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        NUFFTOp
        Sense
        Nc
        N
        Nb
        adjoint=false
    end
    
    methods
        function obj = SenseOp(NUFFTOp,sense,Nb)
            obj.NUFFTOp = NUFFTOp;
            obj.Sense = sense;
            obj.Nc = size(sense,4);
            obj.N = size(sense,1);
            obj.Nb = Nb;
        end  
        
         function out = mtimes(this,z)
            if(this.adjoint)
                out = complex(zeros(this.N,this.N,this.N));
                for i = 1:this.Nc
                    out = out + (this.NUFFTOp'*(z(:,i))).*conj(this.Sense(:,:,:,i));
                end            
            else
                out = complex(zeros(this.Nb,this.Nc));
                for i = 1:this.Nc
                    out(:,i) = this.NUFFTOp*(z.*this.Sense(:,:,:,i));
                end
            end
         end
        
          function res = ctranspose(this)
            this.adjoint = xor(this.adjoint,true);
            res = this;
        end

    end
end



