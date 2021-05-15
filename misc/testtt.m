for i=1:10
load(strcat('fb14_d_slc',num2str(i),'.mat'));
dd(:,:,i)=d;
end
dd1=dd(61:250,111:280,:);