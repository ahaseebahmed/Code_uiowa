function plotfigs(test,x1,y1,x11,y11,mtx,txt)


if(strcmp(txt,'noisy_P82432_'))
    scl1 = [0,0.35];
    scl2 = [0,0.35];
else
    scl1 = [0.07,0.25];
    scl2 = [0.04,0.25];
end
figure(1);imagesc(imrotate(squeeze(test(:,y11,:)),90),scl1);
axis off; axis('image');colormap(gray)
fname = [txt,num2str(x1),num2str(y1),'_',num2str(mtx),'_y.pdf'];
export_fig(fname)

figure(2);imagesc(imrotate(squeeze(test(x11,:,:)),90),scl2);
axis off; axis('image');colormap(gray)
fname = [txt,num2str(x1),num2str(y1),'_',num2str(mtx),'_x.pdf'];
export_fig(fname)

end

