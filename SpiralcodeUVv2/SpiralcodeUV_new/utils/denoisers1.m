function Z=denoisers1(x,W1,W2,b1,b2)
    N=size(x,1);
    x=reshape(x,size(x,1).^2,size(x,3));
    x=x/max(abs(x(:)));
    rr=real(x);
    imm=imag(x);
    rr_i=cat(1,rr,imm);
%     mx_mn=max(rr_i(:))-min(rr_i(:));
%     mn=min(rr_i(:));
%     xx=(rr_i-mn)/mx_mn;
    t1=W1*rr_i'+b1;
    t1(t1<0)=0;
    t2=W2*t1+b2;
    z=t2.*max(abs(x(:)));
    Z=reshape(z,400,size(x,1),2);
    Z=Z(:,:,1)+Z(:,:,2)*1j;
    Z=Z';
    Z=reshape(Z,N,N,400);
    
    
end