
no=[14,15,16,17,22,26];

str='data';
s=[];d=[];


for i=9:9
    load(strcat('CINE_BH_0',num2str(i),'.mat'));
    colormap gray
    y1=bh_data(40:200,40:160,:);
for j=1:size(y1,3)
    d=y1(:,:,j);
    imagesc((d));   

title('draw the endo mask');
h = imfreehand();
endoMask = createMask(h);
%save(strcat('h_bh',num2str(1),'.mat'),'endoMask');
delete h;
area_per_frame(j)=sum(endoMask(:));%*1.56;
end
save(strcat('CINE14_area_slc',num2str(i),'.mat'),'area_per_frame');
area_per_frame=0;
end

