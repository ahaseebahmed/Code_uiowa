function  x = solveUV1( ktraj, w, kdata, csm, V, N, nIterations, SBasis, useGPU)
%function  x = solveUV( ktraj, w, kdata, csm, V, N, nIterations, SBasis, useGPU)
%
% 2d-3d version    CAC 190304

%ktraj_scaled = reshape( ktraj_scaled, [3, nFreqEncoding*ninterleavesPerFrame, numFramesToKeep]);
%w = reshape( w, [nFreqEncoding*ninterleavesPerFrame, numFramesToKeep]);
%kdata = reshape( kdata, [nFreqEncoding*ninterleavesPerFrame, numFramesToKeep, nChannelsToChoose]);

% gpuNUFFT implemented, cpu NUFFT not implemented *** CAC 190306

[nSamplesPerFrame, numFrames, ~] = size( kdata);
[~, nbasis] = size( V);
sTraj = size( ktraj);

% check 3d or 2d
if sTraj(1) == 3 % 3d
    imageDim = 3;
    ktraj_gpu = reshape( ktraj, 3, sTraj(2)*sTraj(3)); %size( ktraj_gpu)
    % w flattening handeled by w(:)
    osf = 1.25; wg = 2.5; sw = 8;
    
    if useGPU
        %w = ones( nSamplesPerFrame*numFrames, 1);
        %w=repmat(dcf,[1 5*numFrames]);
        %FT = gpuNUFFT( ktraj_gpu/N, w(:), osf, wg, sw, [N N N], []);
        %FT = gpuNUFFT( ktraj_gpu/N, ones( nSamplesPerFrame*numFrames, 1), osf, wg, sw, [N N N], []);
        FT = gpuNUFFT( ktraj_gpu/N, w(:), osf, wg, sw, [N N N], []); % why wasn't w passed as argument? *** CAC 190306
        FT2 = gpuNUFFT( ktraj_gpu/N, (w(:) .^ 2), osf, wg, sw, [N N N], []); % squared density needed when adj only.
        Atb = Atb_UV( FT2, kdata, V, csm, N, true, imageDim);
        Reg = @(x) reshape( reshape( x, [N*N*N, nbasis]) * SBasis, [N*N*N*nbasis, 1]);
        AtA = @(x) AtA_UV( FT, x, V, csm, N, nSamplesPerFrame, imageDim) + Reg( x);
    else
        FT = NUFFT( ktraj/N, 1, 0, 0, [N, N, N]);
        Atb = Atb_UV( FT, kdata, V, csm, false, imageDim);
        AtA = @(x) AtA_UV( FT, x, V, csm, nSamplesPerFrame, imageDim);
    end
    x = pcg_quiet( AtA, Atb(:), 1e-5, nIterations); %size( x)
    %x = pcg( AtA, Atb(:), 1e-5, nIterations); %size( x)
    x = reshape( x, [N, N, N, nbasis]); %size( x)
    
else % 2d
    imageDim = 2;
    ktraj_gpu = [real( ktraj(:)), imag( ktraj(:))']
    % w flattening handeled by w(:)
    osf = 2; wg = 3; sw = 8;
    
    if useGPU
        %w = ones( nSamplesPerFrame*numFrames, 1);
        %w=repmat(dcf,[1 5*numFrames]);
        %FT = gpuNUFFT( ktraj_gpu/N, w(:), osf, wg, sw, [N N], []);
        %FT = gpuNUFFT( ktraj_gpu/N, ones( nSamplesPerFrame*numFrames, 1), osf, wg, sw, [N N], []);
        FT = gpuNUFFT( ktraj_gpu/N, w(:), osf, wg, sw, [N N], []);
        FT2 = gpuNUFFT( ktraj_gpu/N, (w(:) .^2), osf, wg, sw, [N N], []);% squared density needed when adj only. 
        Atb = Atb_UV( FT2, kdata, V, csm, N, true, imageDim);
        Reg = @(x) reshape( reshape( x, [N*N, nbasis]) * SBasis, [N*N*nbasis, 1]);
        AtA = @(x) AtA_UV( FT, x, V, csm, N, nSamplesPerFrame, imageDim) + Reg( x);
    else
        FT = NUFFT( ktraj/N, 1, 0, 0, [N, N]);
        Atb = Atb_UV( FT, kdata, V, csm, false, imageDim);
        AtA = @(x) AtA_UV( FT, x, V, csm, nSamplesPerFrame, imageDim);
    end
    x = pcg_quiet( AtA, Atb(:), 1e-5, nIterations);
    %x = pcg( AtA, Atb(:), 1e-5, nIterations);
    x = reshape( x, [N, N, nbasis]);
    
end

end
