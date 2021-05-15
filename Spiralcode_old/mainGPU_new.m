% % 
% % Code for the publication:
% % Sunrita Poddar and Mathews Jacob. "Dynamic MRI using smooThness
% % regularization on manifolds (SToRM)." IEEE transactions on medical 
% % imaging 35.4 (2016): 1106-1115.
% % 
% % Author: Sunrita Poddar, The University of Iowa
% % Date: 24 May 2017
% % 
% % Code solves the l2-SToRM problem described in the above publication.
% % Example here reconstructs a free-breathing ungated cardiac dataset 
% % using a radial trajectory with navigators.
% % 
% % %
% % ==============================================================
% % Set the parameters for the dataset
% % ==============================================================
% % 
% % % % Dimension of the data is: n x n x nf x nc
% % n = 300; % Image dimension is n x n
% % spiral_len = 2496;
% % n_int = 10;
% % nf = 300; % Number of frames
% % lambda = 2e-6; % Regularization parameter
% % coil_ind = 1:10;
% % nc = length(coil_ind); % Number of coils
% % nbasis=30;
N=300;
str_spr=100;
nint=5;
%nf=500;
%nFE=2496;
nCh=4;
nbasis=30;
disp_slice=1;
useGPU = true;
com_pts=100;
ch=[12,21,25,28,24,8,22,5,3,18,11,27,4,34,15,6,14,10,7,16,9,32,26,29,13,20,30,17,33,19];
%% Reconstruction parameters
maxitCG = 15;
alpha = 1e-5;
tol = 1e-6;
display = 0;
osf = 2; wg = 7; sw = 8;
imwidth = N;

% Load data
 load('./../Data/SpiralData_UVirginia/Spiral_cine_3T_1leave_FB_027_full.mat','kdata','ktraj','dcf');
 kdata=squeeze(kdata);
 [nFE,tCh,tspr]=size(kdata);
 nf=floor((tspr-str_spr)/nint);
 kdata=kdata(:,ch(1:nCh),str_spr+1:tspr);
 

 kdata=permute(kdata,[1 3 2]);
 kdata=reshape(kdata,[nFE,nint,nf,nCh]);
 b=kdata(:,:,1:100,:);

% % %
% % ==============================================================
% % Load the data
% % ============================================================== 

% % S.mat: Sampling pattern saved in a cell of size {1 x nf}
% % Example: S{1} contains k-space locations sampled for frame 1
% % These Cartesian locations are obtained by gridding the original
% % % % non-cartesian locations
% % load ('S10.mat'); 
% % S = S(:,1:nf);
% % S = S*n/214;
% % 
% % % b.mat: Acquired data saved in a cell of size {nf x nc}
% % % Example: b{10, 3} contains k-space data acquired by the 3rd coil
% % % at sampling locations in S{10}
% % % This Cartesian data is obtained by gridding the original
% % % non-cartesian data
% % load ('b10.mat');
% % b = b(:,1:nf,:);

k=cell2mat(ktraj);%k=S(:,1:nf);
k=k(:,str_spr+1:tspr);
%k=(k-min(k(:)))./(max(k(:))-min(k(:)))*0.5;
w=repmat(dcf,[1 nint*nf]);%w = sqrt(abs(k));
clear ktraj dcf;
S=1.3*k(:,1:500);
S=reshape(S,[nFE*nint,100]);


% csm.mat: Estimated coil sensitivity maps of size {n x n x nc}
% Maps are obtained using ESPIRIT algorithm
% load ('csm_spiral_27.mat');
% csm = csm(:,:,1:coil_ind);

% % %
% % ==============================================================
% % Compute the weight matrix
% % ============================================================== 

% load ('W_27.mat');
% L=L(1:100,1:100);
% W = W(1:nf,1:nf);
% 
% % Add some small temporal regularization
% t = 0.05*(circshift(eye(nf),[0 1]) + circshift(eye(nf),[1 0]));
% W = max(W,t);
% 
% % Compute the Laplacian matrix as L = D - W
% L = (diag(sum(W,1))-W);
% [~,Sbasis,V]=svd(L);
% V=V(:,end-nbasis+1:end);
% Sbasis=Sbasis(end-nbasis+1:end,end-nbasis+1:end);

% clear L W bCom

%% 
%==============================================================
%Generate NUFFT operators
%============================================================== 

for i=1:1
    i
    % Frame-wise A' operator
    [A1,B1] = creatA(transpose(S(:,i)),n);
    At_fr{i} = @(x) INFT_new(x,A1,B1);
    
    % Frame-wise A'A operator
    [A2, B2] = creatA_zeropadded(transpose(S(:,i)),n,2*n);
    Q(:,:,i) = fft2(fftshift(INFT_new(ones(nFE*nint,1),A2,B2)/n^2));      
end

At = @(z) DFT2_multicoil_At_old(z,At_fr,csm,n,nf,nc);
AtAnew = @(z) AtAgpuBlock_old(z,Q,csm,n,nf,nc);
%clear Q csm

%%
%==============================================================
% Solve the optimization problem: min_X { ||AX-b||^2 + lambda*trace(XLX')}
%============================================================== 
% We use the conjugate gradient algorithm to solve:
% A'A(X) + lambda*XL = A'b
% where A is the forward operator consisting of Fourier under-sampling and
% coil-sensitivity maps.

% Compute c = A'b

% for ii=1:nf
%     bv(:,:,ii)=b(:,ii)*V(ii,:);
% end
% 
% for 
b=reshape(b,[nFE*nint,nf,nc]);
c = At(b(:));

clear b

% Compute the function handle: A'A(X) + lambda*XL
gradX = @(z)(AtAnew(z)+lambda*XL(z,L,n,nf));

% Run the Conjugate Gradient algorithm 
X= pcg(gradX,c(:),1,10);

%%
%==============================================================
% Reshape the images for viewing and save
%============================================================== 

X = abs(reshape(X,n,n,nf));

save('result_new.mat', 'X', '-v7.3');

