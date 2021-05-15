liste = rdir(['./ReconData/','*SAX_gNAV_gre.dat'],'','./ReconData/');
files = {liste.name};
rd=pwd;

for ii=1:length(files)
newdirname = strtok(files{ii},'/');
mkdir(['./Data/',newdirname]);
out=['./Data/',newdirname];
filename = ['./ReconData/',files{ii}];
listt=dir(filename);
cd(filename);
t1=strtok(filename,'/');
for jj=3:length(listt) 

    load(listt(jj).name);
    t2=[out,'/',listt(jj).name];
    cd(rd);
    save(t2,'U1','D','-v7');
    cd(filename);
end
cd(rd)
end