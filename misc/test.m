no=[9,10,11,12,13,14,15,16,17,18,19,20];
no=[28,29,30,31,32,33,34,35,36,37,38];


str='CINE_BH_'; 
for ii=1:length(no)
load(strcat(str,num2str(no(ii)),'.mat'));
d=bh_data(:,:,1);
s=bh_data(:,:,7);


 save(strcat('dia_slc',num2str(no(ii)),'.mat'),'d');
  save(strcat('sys_slc',num2str(no(ii)),'.mat'),'s');

 %save('dia_slc38.mat','d');
 %save('sys_slc20.mat','s');

end
 
 
 dir='SA_CINE_00';
 no=[12,14,15,20,10,13,17,18,19,16,9,11];
 no=[9,10,11,12,13,14,15,16,17,18,19,20];
 no=[9,10,19];

 
 
 
 for i=1:size(no,2)
     dd=[dir,num2str(no(i)),'/','dia_slc',num2str(no(i)),'.mat'];
     load(dd);
     subplot(3,1,i);
     imagesc(d);
     pause(0.5);
 end

  dir='SA_CINE_00';
 no=[12,14,15,20,10,13,17,18,19,16,9,11];
 no=[9,10,11,12,13,14,15,16,17,18,19,20];
 no=[19];
 
 for i=1:size(no,2)
     dd=[dir,num2str(no(i)),'/','sys_slc',num2str(no(i)),'.mat'];
     load(dd);
     subplot(1,1,i);
     imagesc(s);
 end

d=y1(:,:,116);
s=y1(:,:,145);
save('fb_dia_slc20.mat','d');
save('fb_sys_slc20.mat','s');
 
 y = reshape(reshape(x,[N*N,nBasis])*V',[N,N,numFramesToKeep]);

%  
%  dt=d(33:end-32,:);
%  
% [xx,yy]=ndgrid(1:size(dt,1),1:size(dt,2));
% [xx1,yy1]=ndgrid(1:0.56:size(V,1),1:0.56:size(V,2));
% Vq = interpn(xx,yy,ph,ss,V,xx1,yy1,ph1,ss1);
 
 for ii=1:size(y1,3)
     diff(ii)=norm(abs(dt-y1(:,:,ii)),'fro');  
 end
 
 %-------------------Free breathing-------------------%
  N=420;
  nBasis=30;
numFramesToKeep=380;
dir='data16';
 %no=[12,14,15,20,10,13,17,18,19,16,9,11];
 %no=[9,10,11,12,13,14,15,16,17,18,19,20];
 no=[1:9];
 %dia=[132,102,165,102,57,74,113,110];%data 26
 %sys=[150,113,159,120,77,94,152,203];
 %dia=[142,310,310,302,237,254,180,186,106,103];%data 15
 %sys=[130,323,324,312,245,276,209,205,117,98];
  %dia=[182,204,218,98,75,39,123,129];%data 22
 %sys=[193,219,290,108,89,91,134,139];
   dia=[85,215,234,81,126,154,157,176,118];%data 16
 sys=[96,225,245,90,135,244,173,160,135];
 
 
 for i=1:size(no,2)
     in_dir=['resn_16_',num2str(no(i))];
     dd1=[dir,'/','fb1_dia_slc',num2str(no(i)),'.mat'];
     dd2=[dir,'/','fb1_sys_slc',num2str(no(i)),'.mat'];
     load(in_dir);
    y = reshape(reshape(x,[N*N,nBasis])*V',[N,N,numFramesToKeep]);
    y=rot90(flipud(abs(y)),1);
    d=(abs(y(92:300,151:270,dia(i))));
    s=(abs(y(92:300,151:270,sys(i))));
    save(dd1,'d');
    save(dd2,'s');
 end

  dir='SA_CINE_00';
 no=[12,14,15,20,10,13,17,18,19,16,9,11];
 no=[9,10,11,12,13,14,15,16,17,18,19,20];
 no=[19];
 
 for i=1:size(no,2)
     dd=[dir,num2str(no(i)),'/','sys_slc',num2str(no(i)),'.mat'];
     load(dd);
     subplot(1,1,i);
     imagesc(s);
 end

dr=[20,21,22,23,24];
for i=1:size(yy,2)-22
    res(i)=norm(abs(yy(:,i:i+22-1)-yy4),'fro');
end 
 
 