function [kspacecoils]=preinterpolateOneCoil(Xin,Yin,kspace)

sparse_radial_data= squeeze(kspace);
[sx, sy,sz]=size(sparse_radial_data);

cart_k_space_sparse_all=zeros(sx,sx,sz);


for noi=1:sz
    
    img=sparse_radial_data(:,:,noi);
    
    Z=(flipud(img));
    
    X=Xin(:,:,noi);
    Y=Yin(:,:,noi);
    
    XI=round(Xin(:,:,noi));
    YI=round(Yin(:,:,noi));
    warning off
    
    clear ZI
    
    ZI=griddata(X,Y,Z,XI,YI);
    
    KI=zeros(size(ZI));
    ii=find(ZI>0);
    jj=find(ZI<=0);
    
    KI(ii)=ZI(ii);
    KI(jj)=ZI(jj);
    
    clear cart_k_space
    cart_k_space=KI;
    
    cart_k_space_sparse=zeros(sx,sx);
    
    clear temp_X
    clear temp_Y
    
    temp_X=XI+sx/2;
    temp_Y=YI+sx/2;

    
    for i=1:sx
        for j=1:sy
            try
                cart_k_space_sparse(temp_X(i,j),temp_Y(i,j))=cart_k_space(i,j);
            catch
            end
        end
    end
      
    cart_k_space_sparse_all(:,:,noi)=cart_k_space_sparse;
    
end

kspacecoils = cart_k_space_sparse_all;
