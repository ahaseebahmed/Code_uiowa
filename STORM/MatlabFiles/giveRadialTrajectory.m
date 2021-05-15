function [X_all_gr, Y_all_gr]=giveRadialTrajectory(n,n_lines_fr,clin,start,nframes)



X_gr=zeros(n,n_lines_fr);
Y_gr=zeros(n,n_lines_fr);

X_all_gr=zeros(n,n_lines_fr,nframes);
Y_all_gr=zeros(n,n_lines_fr,nframes);

start = start+1;
%% X_gr and Y_gr for golden ratio
%% X_gr and Y_gr for golden ratio

old=mod(((1-clin/n_lines_fr)*(start-1)-1)*2*pi/3.23606797,2*pi);
u=0;
for noi = 1: nframes
    clear X_gr; clear Y_gr;l=1;
    for counter=1:n_lines_fr
        m =  mod(u,n_lines_fr);
        
        if (m<clin)
            theta_radian = pi*m/clin;
        else
            theta_radian = mod(old+2*pi/3.23606797,2*pi);
            old = theta_radian;
            
        end
        
        k=1;
        for r=-(n/2-1):n/2
            [x, y]=pol2cart(theta_radian,r);
            X_gr(k,l)=x;
            Y_gr(k,l)=y;
            k=k+1;
        end
        l=l+1;
        
        u=u+1;
    end
    X_all_gr(:,:,noi)=X_gr;
    Y_all_gr(:,:,noi)=Y_gr;
end
aa=1;
