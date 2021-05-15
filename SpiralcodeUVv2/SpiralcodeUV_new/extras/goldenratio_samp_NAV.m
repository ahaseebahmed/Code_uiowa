function [Samp, NAV] = goldenratio_samp_NAV(N1,N2, Nt, lines, Flag)  

Samp=zeros(N1,N2,Nt);

ang =0;
for frameno=1:Nt,
y=-N1/2:1:N1/2-0.5;  % Create an array of N points between -N/2 and N/2
x=(linspace(-N2/2,N2/2,length(y)));

i=1;

%% NAV lines

if Flag==1
klocn=complex(y*cos(0),x*sin(0));kloc_all(:,1)=klocn;
klocn=complex(y*cos(pi/2),x*sin(pi/2));kloc_all(:,2)=klocn;
klocn=complex(y*cos(pi/4),x*sin(pi/4));kloc_all(:,3)=klocn;
klocn=complex(y*cos(-pi/4),x*sin(-pi/4));kloc_all(:,4)=klocn;
for i = 5: lines
         ang = ang+pi*111.25/180;
         klocn=complex(y*cos(ang),x*sin(ang));
         kloc_all(:,i)=klocn;
end

else
 for i = 1: lines
         klocn=complex(y*cos(ang),x*sin(ang));
         kloc_all(:,i)=klocn;
         %ang = ang+pi/((sqrt(5)+1)/2);
         ang = ang-pi/((sqrt(5)+1)/2);
end   
end   
     % Round the collected data to the nearest cartesian location   
     kcart=round(kloc_all+(0.5+0.5*1i));

    % plot(kcart,'*');title('k locations after nearest neighbor interpolation: Center (0,0)');
%     
%     
    % Next, shift the cartesian locations accordingly such that the center
    % is now at (N/2,N/2); {Previously the center in kcart was (0,0)}
    kloc1 = round(kcart)+((N1/2+1)+(N2/2+1)*1i);
    kloc1real = real(kloc1); kloc1real = kloc1real - N1*(kloc1real>N1);
    kloc1imag = imag(kloc1); kloc1imag = kloc1imag - N2*(kloc1imag>N2);
    kloc1real = kloc1real + N1*(kloc1real<1);
    kloc1imag = kloc1imag + N2*(kloc1imag<1);
    kloc1 = kloc1real + 1i*kloc1imag;
  %% Use this when the radial lines = 24 per frame
    % Fill the data into a sqaure matrix of size (N x N); 
    % Subsequently, the filling is done for all the frames
%     for i=1:size(kloc1,1)
%         for j=1:size(kloc1,2)
%         D(real(kloc1(i,j)),imag(kloc1(i,j)),frameno)=  Data(i,j)*1e6;
%         Samp(real(kloc1(i,j)),imag(kloc1(i,j)),frameno) = 1;
%         
%         end

%     end
kloc2 = kloc1; 

%% Use this for undersampling [For radial lines less than 24 per frame]
%step = mod(frameno,4);
%if step == 0, step = 4; end

%kloc2 = kloc1(:,step:4:end);

%randpick = round(1 + (72 - 1)*rand(1,24));

 
 %for ii = 1: 24, kloc2(:,ii) = kloc1(:,randpick(ii)); end

for i =1:size(kloc2,1)
    for j = 1:size(kloc2,2)
        %D_us(real(k_us(i,j)),imag(k_us(i,j)),frameno) = Data(i,j)*1e6; 
        Samp(real(kloc2(i,j)),imag(kloc2(i,j)),frameno) = 1;
    end
end
end;
NAV=zeros(size(Samp(:,:,1)));
for i =1:size(kloc2,1)
    for j = 1:4
        NAV(real(kloc2(i,j)),imag(kloc2(i,j)))=1;
    end
end

end