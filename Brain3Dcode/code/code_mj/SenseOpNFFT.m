classdef SenseOpNFFT
    %UNTITLED Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        NUFFTplan
        csm
        Nc = 1;
        N
        Nb
        sense = false;
        adjoint=false
    end
    
    methods
        function obj = SenseOpNFFT(N,Nb,csm,k)
            fprintf('Number of threads: %d\n\n', nfft_get_num_threads());
            if(exist(csm))
                obj.Nc = size(csm,4);
                obj.csm = reshape(csm,N*N*N,obj.Nc);
                obj.sense = true;
            end
            obj.N = N;
            obj.Nb = Nb;
            obj.NUFFTplan=nfft(3,N,Nb); % create plan of class type nfft
            if(nargin==4)
                obj.NUFFTplan.x=k;
            end
        end         
        function obj = setKspaceTraj(obj,k)
            obj.NUFFTplan.x=k;
        end
            
         function out = mtimes(obj,z)
            if(obj.adjoint)
                out = complex(zeros(prod(obj.N),1));
                if(obj.sense)
                    for i = 1:obj.Nc
                        obj.NUFFTplan.f=z(:,i);
                        nfft_adjoint(obj.NUFFTplan);
                        out = out + obj.NUFFTplan.fhat.*obj.csm(:,i);
                    end     
                else
                    obj.NUFFTplan.f=z(:);
                    out = reshape(obj.NUFFTplan.fhat,obj.N);
                end
            else
                out = complex(zeros(obj.Nb,obj.Nc));
                for i = 1:obj.Nc
                    coil = z.*obj.csm(:,:,:,i);
                    obj.plan.f= coil(:);
                    nfft_trafo(obj.NUFFTplan);
                    out(:,i) = obj.NUFFTplan.fhat;
                end
            end
         end
        
          function res = ctranspose(obj)
            obj.adjoint = xor(obj.adjoint,true);
            res = obj;
        end

    end
end



