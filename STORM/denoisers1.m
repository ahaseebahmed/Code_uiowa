function R=denoisers1(x,W1,W2,b1,b2,gw)
    n=512;
    nf=200;
   
    aa=max(abs(x(:)));
    x=x./aa;
    x=reshape(x,n.^2,nf);
    rr=real(x);
    imm=imag(x);
    xx=cat(1,rr,imm);
    %mx_mn=max(rr_i(:))-min(rr_i(:));
    %mn=min(rr_i(:));
    %xx=(rr_i-mn)/mx_mn;
    %% -------------------------
    t1=W1*xx'+b1;
    t1(t1(:)<0)=0;
    t2=W2*t1+b2;
    t2(t2(:)<0)=0;
    t3=W3*t2+b3;
    t3(t3(:)<0)=0;
    t4=W4*t3+b4;
    t4(t4(:)<0)=0;
    t5=W5*t4+b5;
    t5(t5(:)<0)=0;
    t6=W6*t5+b6;
    %% ----------------------------
    z=t6;
    Z=reshape(z,200,512,512,2);
    Z=Z(:,:,:,1)+Z(:,:,:,2)*1j;
    Z=permute(Z,[2,3,1]);
    Z=Z*aa;
    x=x*aa;
    x=reshape(x,n,n,nf);
    %Z=scale(Z,x);
    mn=median(Z,3);
    %gw=gw.*mn;
    %gw=(1-gw)+1e-5;
    %gw(gw(:)<=1e-12)=1;
    %Z=Z.*gw;
    R=(x(:)-Z(:));
    R=reshape(R,512,512,200);
    %Z=Z/max(abs(Z(:)));
    R=R(:);
    
end

%     tmp=reshape(t1,10,512,512,2);
%     gw = repmat(gw,[1,1,20]);
%     gw=reshape(gw,[512,512,10,2]);
%     gw=permute(gw,[3,1,2,4]);
%     tmp1=tmp.*gw;
%     t1=reshape(tmp1,[10,512*512*2]);


% 
% t11=reshape(t1,[10,512,512,2]);
% tt=t11(:,:,:,1);
% tt=fftshift(fftshift(tt,2),3);
% tt0=t11(:,:,:,2);
% tt0=fftshift(fftshift(tt0,2),3);
% tt=[tt tt0];