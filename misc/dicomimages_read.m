clear all;

in=[10,11,12,13,14,15,16,17,18,19,20,21];
in=[9];
in=[28,29,30];
in=[31,32,33,34,35,36,37,38,39];



for n=1:length(in)
list = dir(strcat('*0',num2str(in(n)),'/*.IMA'));
files = {list.name};
for i=1:length(files)
    file_name=files{i};
    bh_data(:,:,i)=double(rot90(dicomread([list(1).folder,'/',file_name])));
end
save(strcat('CINE_BH_',num2str(in(n)),'.mat'),'bh_data');
end