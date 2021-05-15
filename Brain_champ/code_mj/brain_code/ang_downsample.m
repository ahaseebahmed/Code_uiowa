function [keepg,keepdir]=ang_downsample(g,init,thr,checkcount)
g0=g(init,:);
count=1;
keepg=g0;
keepdir=init;
for j=1:length(g)
    [sx sy]=size(keepg);
    ang=abs(acos(g(j,:)*keepg'))*180/pi;
    %if(all(ang>thr))
    if( all ( (ang>thr) + (ang>(180-thr)) ))
        keepg=cat(1,keepg,g(j,:));
        keepdir=cat(1,keepdir,j);
        count=count+1;
      end
end
if (checkcount>count)
    disp('Decrease threshold');
elseif(checkcount<count)
    disp('Increase threshold');
end
