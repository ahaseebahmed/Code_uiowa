function out = mattogif(filename,y1)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here%%%15, 18 ..... 138,146
out=[];

    y1=(y1-min(y1(:)))./(max(y1(:))-min(y1(:)));
    y1=y1.*(255);
    map=colormap(gray(255));
for i = 1:(size(y1,3))
 
if i == 1      
    imwrite((y1(:,:,i)),map,filename,'gif','LoopCount',Inf,'DelayTime',0.04);
else  
    imwrite((y1(:,:,i)),map,filename,'gif','WriteMode','append','DelayTime',0.04);
end
end


