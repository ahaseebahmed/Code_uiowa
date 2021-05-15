
str='slice_';
aa=[1,12];
for j=3:3
str1=strcat(str,num2str(j),'.mat');
%load(str1);
for i=2:2%size(c_cyc_exp,3)
    colormap jet;
imagesc(rot90(abs(double(c_cyc_exp(:,:,aa(i),j))),0))%%

title('draw the endo mask');
h = imfreehand();
LV = createMask(h);
%save(strcat('h_bh',num2str(1),'.mat'),'endoMask');
delete h;
h = imfreehand();
RV = createMask(h);
%save(strcat('h_bh',num2str(1),'.mat'),'endoMask');
delete h;
%area_per_frame(i)=sum(endoMask(:));%*1.56;
cc1=bwboundaries(single(LV));
cc2=bwboundaries(single(RV));

con1=cc1{1};
con2=cc2{1};

save(strcat('res_exp_',num2str(i),'_',str1),'con1','con2');
end
%save(strcat('res_inp_',str1),'area_per_frame');

end

figure; hold on;
str1='res_inp_2_slice_';
str3='res_exp_2_slice_';

for j=3:7
str2=strcat(str1,num2str(j),'.mat');
load(str2);
%so_area=so_area+(area_per_frame);
plot3(smooth(con1(:,1)),smooth(con1(:,2)),repmat(j,size(con1,1)),'b','LineWidth',1);
hold on;
plot3(smooth(con2(:,1)),smooth(con2(:,2)),repmat(j,size(con2,1)),'b','LineWidth',1);

str4=strcat(str3,num2str(j),'.mat');
load(str4)
plot3(smooth(con1(:,1)),smooth(con1(:,2)),repmat(j,size(con1,1)),'color',[0.9100    0.4100    0.1700],'LineWidth',1);
hold on;
plot3(smooth(con2(:,1)),smooth(con2(:,2)),repmat(j,size(con2,1)),'color',[0.9100    0.4100    0.1700],'LineWidth',1);
%save(('res_exp_soa.mat'),'so_area');
end


figure; hold on;
so_area=zeros(16,1);
str='res_inp_slice_';
for j=1:8
str1=strcat(str,num2str(j),'.mat');
load(str1);
so_area=so_area+(area_per_frame(:));
%plot(area_per_frame)
save(('res_inp_soa.mat'),'so_area');
end
