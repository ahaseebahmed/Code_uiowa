function [x, y, z] = computePhyllotaxis (N, nseg, nshot, flagSelfNav, flagPlot)


[polarAngle, azimuthalAngle, vx, vy, vz] = phyllotaxis3D (nshot, nseg, flagSelfNav);

r = (-0.5 : 1/N : 0.5-(1/N));
azimuthal  = repmat(azimuthalAngle,[N 1]);
polar = repmat(pi/2-polarAngle,[N 1]);

R = repmat(r',[1 nshot*nseg]); 

[x, y, z] = sph2cart(azimuthal,polar,R);


x = reshape(x, [N, nseg, nshot]);
y = reshape(y, [N, nseg, nshot]);
z = reshape(z, [N, nseg, nshot]);


%% ... plot 

if flagPlot

    figure
    for shot = 1%:nshot%1:20 
        for seg = 1:nseg
            plot3(squeeze(x(:,seg,shot)),squeeze(y(:,seg,shot)),squeeze(z(:,seg,shot)))
            title(['N = ',num2str(N),'  nseg = ',num2str(nseg),'  nshot = ',num2str(nshot)])
            hold on

                if seg==1
                   hold on
                   plot3(x(end,seg,shot),...
                         y(end,seg,shot),...
                         z(end,seg,shot),'.-k','linewidth',2)
                else
                   plot3([x(end,seg-1,shot),x(end,seg,shot)],...
                         [y(end,seg-1,shot),y(end,seg,shot)],...
                         [z(end,seg-1,shot),z(end,seg,shot)],'.-k','linewidth',2)
                end
            axis([-.5 .5 -.5 .5 -.5 .5])

            pause(.1)

        end
        pause(.3)
    end
    hold off
end


