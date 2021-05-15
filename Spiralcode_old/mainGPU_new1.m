% 
% % Code for the publication:
% % Sunrita Poddar and Mathews Jacob. "Dynamic MRI using smooThness
% % regularization on manifolds (SToRM)." IEEE transactions on medical 
% % imaging 35.4 (2016): 1106-1115.
% 
% % Author: Sunrita Poddar, The University of Iowa
% % Date: 24 May 2017
% 
% % Code solves the l2-SToRM problem described in the above publication.
% % Example here reconstructs a free-breathing ungated cardiac dataset 
% % using a radial trajectory with navigators.
% 
% %%
% %==============================================================
% % Set the parameters for the dataset
% %==============================================================
% 
% % Dimension of the data is: n x n x nf x nc
% n = 300; % Image dimension is n x n
% spiral_len = 2496%%2048;%2496;
% n_int = 100;
% nf = 30; % Number of frames
% lambda = 2e-6; % Regularization parameter
% coil_ind =[12,21,25,28,24,8,22,5,3,18,11,27,4,34,15,6,14,10,7,16,9,32,26,29,13,20,30,17,33,19];
% nc = 10; % Number of coils
% nff=1;
% %%
% %==============================================================
% % Load the data
% %============================================================== 
% 
% % S.mat: Sampling pattern saved in a cell of size {1 x nf}
% % Example: S{1} contains k-space locations sampled for frame 1
% % These Cartesian locations are obtained by gridding the original
% % non-cartesian locations
% % load ('S10.mat'); 
% % S=S(:,1:10);
% % S = reshape(S,[2048*100,1]);
% % S = S*n/214;
% % % 
% % % b.mat: Acquired data saved in a cell of size {nf x nc}
% % % Example: b{10, 3} contains k-space data acquired by the 3rd coil
% % % at sampling locations in S{10}
% % % This Cartesian data is obtained by gridding the original
% % % % non-cartesian data
% % load ('b10.mat');
% % 
% % b = b(:,1:10,1:nc);
% % b=reshape(b,[2048*100,1,nc]);
% % 
% % % csm.mat: Estimated coil sensitivity maps of size {n x n x nc}
% % % Maps are obtained using ESPIRIT algorithm
%  load ('coilImages.mat');
%  % load ('csm300.mat');
% 
%  csm = csm(:,:,1:nc);
% 
% 
% load('Spiral_cine_3T_1leave_FB_027_full.mat','kdata','ktraj','dcf');
% b=squeeze(kdata);
% b=permute(b,[1, 3, 2]);
% b=b(:,:,coil_ind(1:nc));
% b=reshape(b,[spiral_len*n_int,nf,nc]);
% b=b(:,1:nff,:);
% S=cell2mat(ktraj);
% S=reshape(S,[spiral_len*n_int,nf]);
% S=S(:,1:nff);
% w=repmat(dcf,[1 n_int]);

%%
%==============================================================
% Compute the weight matrix
%============================================================== 
% 
% load ('W10.mat');
% W = W(1:nf,1:nf);
% 
% % Add some small temporal regularization
% t = 0.05*(circshift(eye(nf),[0 1]) + circshift(eye(nf),[1 0]));
% W = max(W,t);
% 
% % Compute the Laplacian matrix as L = D - W
% L = sparse(diag(sum(W,1))-W);
% 
% clear W bCom

%% 
%==============================================================
%Generate NUFFT operators
%============================================================== 
% 
for i=1:nff
    i
    % Frame-wise A' operator
    [A1,B1] = creatA(transpose(S(:,i)),n);
    At_fr{i} = @(x) INFT_new(x,A1,B1);
    
    % Frame-wise A'A operator
    [A2, B2] = creatA_zeropadded(transpose(S(:,i)),n,2*n);
    Q(:,:,i) = fft2(fftshift(INFT_new(ones(spiral_len*n_int,1),A2,B2)/n^2));      
end

At = @(z) DFT2_multicoil_At(z,At_fr,csm,n,nff,nc);
AtAnew = @(z) AtAgpuBlock(z,Q,csm,n,nff,nc);
% % % %clear Q csm
% % 
% % %%
% % %==============================================================
% % % Solve the optimization problem: min_X { ||AX-b||^2 + lambda*trace(XLX')}
% % %============================================================== 
% % % We use the conjugate gradient algorithm to solve:
% % % A'A(X) + lambda*XL = A'b
% % % where A is the forward operator consisting of Fourier under-sampling and
% % % coil-sensitivity maps.
% % 
% % % Compute c = A'b
% b=bsxfun(@times,b,w(:));
 c = At(b);
% clear kdata;
% clear b

% Compute the function handle: A'A(X) + lambda*XL
gradX = @(z)(AtAnew(z));%+lambda*XL(z,L,n,nf));

% Run the Conjugate Gradient algorithm 
X= pcg(gradX,c(:),10^-10,70);

%%
%==============================================================
% Reshape the images for viewing and save
%============================================================== 

X = abs(reshape(X,n,n,nff));

save('result.mat', 'X', '-v7.3');

