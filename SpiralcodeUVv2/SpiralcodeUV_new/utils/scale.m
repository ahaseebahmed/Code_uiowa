
function[U]= scale(U,I)
U = (U); I = (I);
for ii=1:size(U,3)
alpha = sum(dot(U(:,:,ii),I(:,:,ii)))/(sum(dot(U(:,:,ii),U(:,:,ii))));

U(:,:,ii)=(alpha)*U(:,:,ii);
end

