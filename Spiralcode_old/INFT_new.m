function x=INFT_new(X,A2,B2,w)

[N1,N2] = size(B2);
AX = zeros(N1,N2);

for i=1:N2
    AX(:,i) = X(:).*conj(A2(:,i));
end
if (nargin<4)
    x = AX.'*conj(B2);
else
    B3=bsxfun(@times,conj(B2),w);
    x=AX.'*B3;
end



