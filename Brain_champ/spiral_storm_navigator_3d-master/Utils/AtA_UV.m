function y = AtA_UV( FT, x, V, csm, N, NsamplesPerFrame, imageDim,w)
%function y = AtA_UV( FT, x, V, csm, N, NsamplesPerFrame, imageDim)
%
% 2d-3d version     CAC 190304

[nFrames, nBasis] = size( V);

%imageDim % debugging

if imageDim == 3 % 3d
    [~, ~, ~, nChannelsToChoose] = size( csm);
    x = reshape( x, [N, N, N, nBasis]);
    
    y = zeros( N, N, N, nBasis);
    
    for j = 1:nChannelsToChoose
        virtualKspace = zeros( NsamplesPerFrame, nFrames);
        for i = 1:nBasis
            temp = FT * (x(:, :, :, i) .* csm(:, :, :, j));
            virtualKspace = virtualKspace + reshape( temp, [NsamplesPerFrame, nFrames]) * diag( V(:, i));
        end
        virtualKspace=virtualKspace.*w;
        for i = 1:nBasis
            temp = virtualKspace * diag( V(:, i));
            y(:, :, :, i) = y(:, :, :, i) + (FT' * temp(:)) .* conj( csm(:, :, :, j));
        end
    end
    
    y = y(:);
    
else %2d
    [~, ~, nChannelsToChoose] = size( csm);
    x = reshape( x, [N, N, nBasis]);
    
    y = zeros(N, N, nBasis);
    
    for j = 1:nChannelsToChoose
        virtualKspace = zeros( NsamplesPerFrame, nFrames);
        for i = 1:nBasis
            temp = FT * (x(:, :, i) .* csm(:, :, j));
            virtualKspace = virtualKspace + reshape( temp, [NsamplesPerFrame, nFrames]) * diag( V(:, i));
        end
        for i = 1:nBasis
            temp = virtualKspace * diag( V(:, i));
            y(:, :, i) = y(:, :, i) + (FT' * temp(:)) .* conj( csm(:, :, j));
        end
    end
    
    y = y(:);
    
end

end

