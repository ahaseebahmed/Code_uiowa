function Atb = Atb_UV( FT, kdata, V, csm, N, useGPU, imageDim)
%function Atb = Atb_UV( FT, kdata, V, csm, N, useGPU, imageDim)
%
% 2d-3d version     CAC 190304

[~, nBasis] = size( V);

%max( abs( conj( csm(:, :, :, 1))), [], 'all') % debugging

if imageDim == 3 % 3d
    [~, ~, ~, nChannelsToChoose] = size( csm);
    Atb = zeros( N, N, N, nBasis);
    
    if useGPU
        for i = 1:nBasis
            for j = 1:nChannelsToChoose
                temp = squeeze( kdata(:, :, j)) * diag( V(:, i)); % size( temp(:))
                Atb(:, :, :, i) = Atb(:, :, :, i) + (FT' * temp(:)) .* conj( csm(:, :, :, j)); %max( abs( Atb), [], 'all')
            end
        end
    else %cpu
        for i = 1:nBasis
            for j = 1:nChannelsToChoose
                temp = squeeze( kdata(:, :, j)) * diag( V(:, i));
                temp = reshape( temp, [ 2496, 4, 450]); % <<_________*********** CAC 190403
                Atb(:, :, :, i) = Atb(:, :, :, i) + (FT' * temp) .* conj( csm(:, :, :, j));
            end
        end
    end
    
else %2d
    [~, ~, nChannelsToChoose] = size( csm);
    Atb = zeros( N, N, nBasis);
    
    if useGPU
        for i = 1:nBasis
            for j = 1:nChannelsToChoose
                temp = squeeze( kdata(:, :, j)) * diag( V(:, i));
                Atb(:, :, i) = Atb(:, :, i) + (FT' * temp(:)) .* conj( csm(:, :, j));
            end
        end
    else %cpu
        for i = 1:nBasis
            for j = 1:nChannelsToChoose
                temp = squeeze( kdata(:, :, j)) * diag( V(:, i));
                temp = reshape( temp, [2496, 4, 450]); % <<_________*********** CAC 190403
                Atb(:, :, i) = Atb(:, :, i) + (FT' * temp) .* conj( csm(:, :, j));
            end
        end
    end
    
end





