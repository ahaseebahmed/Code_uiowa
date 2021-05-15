
addpath('./../');
pre_cc=25;
no_slice=10;

%%---------Reading data----------------
load(strcat('res_0.025_4.5_0.3_',num2str(4),'.mat'));
y=abs(rot90(flipud(y)));
y=round((y-min(y(:)))./(max(y(:)-min(y(:))))*255);
y1=y(N/2-N/4+1:N/2+N/4,N/2-N/4+1:N/2+N/4,:);

% for i=1:size(y,3)
%     imagesc(abs(y(:,:,i)));colormap gray;axis off;pause()    
% end

imagesc(y1(:,:,10))%%
title('draw the line to get temporal profile');
h = imline();
%pos = createMask(h);
pos=getPosition(h);
imagesc(squeeze(y1(round(pos(1,2)),:,:)));%%
hold on;
plot(N/8+N*0.125*V(:,end-1),'r-');
title('draw two points to identify the cardiac cycle at expiration');
h1 = impoint();
pt1=getPosition(h1);
h2 = impoint();
pt2=getPosition(h2);
y1_exp=y1(:,:,pt1(1,1):pt2(1,1));

title('draw two points to identify the cardiac cycle at inpiration');
h3 = impoint();
pt3=getPosition(h3);
h2 = impoint();
pt4=getPosition(h4);
y1_inp=y1(:,:,pt3(1,1):pt4(1,1));


%%%%%%%%%%%%%%%picking Reference frame%%%%%%%%%
% src1=y2(:,:,196);%----end of Inspiration----%slice 5
% src2=y2(:,:,155);%----end of Expiration----
src1=y1(:,:,48);%----end of Expiration----% slice 4
src2=y1(:,:,170);%----end of Inspiration---

%%%%%%%%%%%%%%%Extracting Cardiac Cycle%%%%%%%%%%%%
for ii=1:pre_cc
    si1(ii)=norm(abs(y1(:,:,48+ii)-src1),'fr');
    mi1(ii)=MI_GG(abs(src1),abs(y1(:,:,48+ii)));
end

cc1=y1(:,:,48:48+15);
for jj=1:pre_cc
    si2(jj)=norm(abs(y1(:,:,170+jj)-src2),'fr');
    mi2(jj)=MI_GG(abs(src2),abs(y1(:,:,170+jj)));    
    
end

%cc1=y1(:,:,170:170+15);
%car_cyc=16;
cc2=y1(:,:,170:170+15);
car_cyc=15;
% for i=1:15
% subplot(1,15,i);imagesc(squeeze(abs(cc2(:,:,i))));colormap gray;
% end

clear si1 si2 mi1 mi2;
%%%%%%%%%%%%%%%%Finding cardiac cycle all slices%%%
for ij=1:no_slice
load(strcat('res_0.025_4.5_0.3_',num2str(ij),'.mat'));
%y=abs(y);
y=abs(rot90(fliplr(y),-1));
y=round((y-min(y(:)))./(max(y(:)-min(y(:))))*255);
y=abs(y(130:210,130:210,:));
%y=abs(y(144:191,154:195,:));

% y=imgaussfilt(y);
% kernel = -1*ones(3)./9;
% kernel(2,2) = 8/3;
% y = round(imfilter(y, kernel));

for ii=1:size(y,3)-car_cyc-10
    tmp=y(:,:,ii:ii+car_cyc);
    si1(ij,ii)=norm(abs(tmp(:)-cc1(:)));
    mi1(ij,ii)=MI_GG(abs(cc1(:)),abs(tmp(:)));    
end

for jj=1:size(y,3)-car_cyc-10
    tmp1=y(:,:,jj:jj+car_cyc);
    si2(ij,jj)=norm(abs(tmp1(:)-cc2(:)));
    mi2(ij,jj)=MI_GG(abs(cc2(:)),abs(tmp1(:)));    
end
end
% 
for i=1:no_slice
subplot(10,1,i);plot(squeeze((si1(i,:))));
end

%%%%%%%%%%%%%%%%finding minimum point%%%%%%%%%
for kk=1:no_slice
    [~,in1(kk)]=min(si1(kk,:));
    [~,in2(kk)]=min(si2(kk,:));
end
for sl=1:10
    load(strcat('res_0.025_4.5_0.3_',num2str(sl),'.mat'));

for mm=1:19
     mn1(sl,mm)=norm(abs(y(:,:,in1(sl))-y(:,:,in1(sl)+mm)));
     mn2(sl,mm)=norm(abs(y(:,:,in2(sl))-y(:,:,in2(sl)+mm)));

end
end

for kk=1:no_slice
    [~,in3(kk)]=min(mn1(kk,4:end));
    [~,in4(kk)]=min(mn2(kk,4:end));
end

for kk=1:no_slice
    c_cyc_inp{kk}=y(:,:,in1(kk):in1(kk)+in3(kk));
    c_cyc_exp{kk}=y(:,:,in2(kk):in2(kk)+in4(kk));
    
end

%%%%%%%%%%%%%%%
clear si1 si2;
for ij=1:no_slice
load(strcat('res_0.025_4.5_0.3_',num2str(ij),'.mat'));
y=abs(rot90(fliplr(y),-1));
y=round((y-min(y(:)))./(max(y(:)-min(y(:))))*255);
y=abs(y(130:195,130:195,:));
%y=abs(y(144:191,154:195,:));

% aa=y(:,:,in1(ij));
% bb=y(:,:,in2(ij));
% for kk=1:pre_cc
%     si1(kk)=norm(abs(aa-y(:,:,in1(ij)+kk)),'fr');
%     si2(kk)=norm(abs(bb-y(:,:,in2(ij)+kk)),'fr');
% end
% [~,cind1]=min(si1);
% [~,cind2]=min(si2);
% c_cyc_inp(:,:,:,ij)=y(:,:,in1(ij):cind1-1);
% c_cyc_exp(:,:,:,ij)=y(:,:,in2(ij):cind2-1);
c_cyc_inp(:,:,:,ij)=y(:,:,in1(ij):in1(ij)+car_cyc);
c_cyc_exp(:,:,:,ij)=y(:,:,in2(ij):in2(ij)+car_cyc);
end
%%%%%%%%%%%%%%Image Registration%%%%%%%%%%%%%%%
c_cyc_inp=abs(rot90(fliplr(c_cyc_inp),-1));
c_cyc_inp=c_cyc_inp(130:195,130:195,:,:);
%y=round((y-min(y(:)))./(max(y(:)-min(y(:))))*255);


cc_inp_reg=abs(c_cyc_inp);
[optimizer, metric] = imregconfig('multimodal');
optimizer.InitialRadius = 0.0004;
optimizer.Epsilon = 1.5e-5;
optimizer.GrowthFactor = 1.05;
optimizer.MaximumIterations = 300;

for i=1:size(c_cyc_inp,4)
    for j=1:size(c_cyc_inp,3)
    moving=abs(c_cyc_inp(:,:,j,i));
    fixed=abs(c_cyc_inp(:,:,j,5));
    cc_inp_reg(:,:,j,i) = imregister(moving,fixed, 'affine', optimizer, metric);
    end
end

%%%%%%%%%%%%%%Image Series Interpolation%%%%%%%%%%
M=size(c_cyc_inp,1);N=size(c_cyc_inp,2);
factr=2;
 no_int_frames=size(c_cyc_inp,4)*factr;
% no_fr=zeros(1,no_int_frames);
% fr = 1:factr:no_int_frames;    % frame time
% no_fr(fr)=1;
% [~,fr_int]=find(no_fr==1);
% Ii = zeros(size(c_cyc_inp,1),size(c_cyc_inp,2),size(c_cyc_inp,3),length(fr_int)) ;   
% 
% for i =1:size(c_cyc_inp,1)
%     for j = 1:size(c_cyc_inp,2)
%             for k = 1:size(c_cyc_inp,3)
%                 I_ijk = squeeze(c_cyc_inp(i,j,k,:)) ;
%                 Iijk = interp1(fr,I_ijk,fr_int,'spline') ;        
%                 Ii(i,j,k,:) = reshape(Iijk,1,1,1,length(fr_int)) ;
%             end
%     end
% end
% c_cyc_inp_new=zeros(M,N,size(c_cyc_inp,3),no_int_frames);
% c_cyc_inp_new(:,:,:,1:2:no_int_frames)=c_cyc_inp;
% c_cyc_inp_new(:,:,:,2:2:no_int_frames)=Ii;

% V=abs(rot90(fliplr(c_cyc_inp),-1));
% V=V(130:195,130:195,:,:);
V=c_cyc_inp;
[xx,yy,ph,ss]=ndgrid(1:size(V,1),1:size(V,2),1:car_cyc+1,1:10);
[xx1,yy1,ph1,ss1]=ndgrid(1:size(V,1),1:size(V,1),1:car_cyc+1,1:0.125:10);
Vq = interpn(xx,yy,ph,ss,V,xx1,yy1,ph1,ss1);

U=c_cyc_exp;
[xx,yy,ph,ss]=ndgrid(1:size(U,1),1:size(U,2),1:car_cyc+1,1:10);
[xx1,yy1,ph1,ss1]=ndgrid(1:size(U,1),1:size(U,1),1:car_cyc+1,1:0.125:10);
Uq = interpn(xx,yy,ph,ss,U,xx1,yy1,ph1,ss1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%--------------CardiacInterpolation
Vc=c_cyc_inp;
[xx,yy,ph,ss]=ndgrid(1:size(Vc,1),1:size(Vc,2),1:car_cyc+1,1:10);
[xx1,yy1,ph1,ss1]=ndgrid(1:size(Vc,1),1:size(Vc,1),1:0.5:car_cyc+1,1:10);
Vqc = interpn(xx,yy,ph,ss,Vc,xx1,yy1,ph1,ss1);

Uc=c_cyc_exp;
[xx,yy,ph,ss]=ndgrid(1:size(Uc,1),1:size(Uc,2),1:car_cyc+1,1:10);
[xx1,yy1,ph1,ss1]=ndgrid(1:size(Uc,1),1:size(Uc,1),1:0.5:car_cyc+1,1:10);
Uqc = interpn(xx,yy,ph,ss,Uc,xx1,yy1,ph1,ss1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i=1:no_slice
subplot(8,1,i);imagesc(squeeze(abs(c_cyc_inp(:,:,1,i))));colormap gray;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i=1:no_slice
subplot(8,1,i);imagesc(squeeze(abs(Vq(:,:,1,i))));colormap gray;
end

%%%%%%%%%%%%%%%%%%%Cancatening the slices%%%%%%%%%%%
res1=[];
res2=[];
for i=1:37%(no_slice*factr)-1
    res1=cat(3,res1,Vq(:,:,:,i));
    %res2=cat(3,res2,Uq(:,:,:,i));
end
res1=abs(rot90(fliplr(res1),-1));
res2=abs(rot90(fliplr(res2),-1));

res3=[];
res4=[];
for i=1:car_cyc+1
    res3=cat(3,res3,squeeze(Vq(:,:,i,:)));
end
for i=1:21
    res4=cat(3,res4,squeeze(c_cyc_exp(:,:,i,:)));
end
res3=abs(rot90(fliplr(res3),-1));
res3=res3(130:195,130:195,:);
res4=abs(rot90(fliplr(res4),-1));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tmpp1=reshape(cc_inp_reg,[size(cc_inp_reg,1),size(cc_inp_reg,2),size(cc_inp_reg,3)*size(cc_inp_reg,4)]);
tmpp=reshape(c_cyc_inp,[size(c_cyc_inp,1),size(c_cyc_inp,2),size(c_cyc_inp,3)*size(c_cyc_inp,4)]);
tmpp=reshape(Vq,[size(Vq,1),size(Vq,2),size(Vq,3)*size(Vq,4)]);
tmpp=reshape(Uq,[size(Uq,1),size(Uq,2),size(Uq,3)*size(Uq,4)]);


for j=1:size(tmpp,3)
   %subplot(10,1,i);plot(si1(i,:));
   subplot(no_slice,car_cyc+1,j);colormap gray; imagesc(abs(c_cyc_inp(:,:,j)));
end
    


