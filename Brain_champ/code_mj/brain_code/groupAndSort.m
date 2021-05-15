function [thetaSortedSub,phiSortedSub] = groupAndSort(theta,phi)

[Nsamples,steps] = size(phi);
theta = pi*theta(:)'/180;
phi = pi*phi(:)'/180;
V = ([sin(theta).*cos(phi); sin(theta).*sin(phi); cos(theta)]);
eps = 1e-2;s=1;
eps = gpuArray(eps); 

indices = zeros(Nsamples,1);
indices(1)=1;

thetaSortedSub = zeros(Nsamples,steps);
phiSortedSub = zeros(Nsamples,steps);
       
for step = 1:steps,
       D = zeros(Nsamples,size(V,2));
       D = gpuArray(D); s = gpuArray(s);V = gpuArray(V);
       indices = gpuArray(indices);
        
       D(1,:) = real(acos(complex(V(:,indices(1))'*V)));

        % find furthest separated samples
        for i=1:Nsamples-1,
            Q = D(1:i,:);
            U=1./(eps + Q.^s);
            [minval,index] = min(max(U,[],1)); 
            indices(i+1) = index;
            temp = real(acos(complex(V(:,index)'*V)));
            D(i+1,:) = (temp);
        end
        %saving the values
        Vsub = V(:,indices);
        phiSub = phi(indices);
        thetaSub = theta(indices);

        % Reducing the size of the original array
        temp = ones(size(V,2),1);
        temp(indices)=0;
        keepindices = find(temp);
        V = gpuArray(V(:,keepindices));
        phi = phi(keepindices);
        theta = theta(keepindices);
        
        % sort them in order

        D = real(acos(complex(Vsub'*Vsub)));
        D(1:Nsamples+1:end)=10;
        indices = gpuArray(zeros(Nsamples,1));
        indices(1)=1;
        lastindex = 1;
        D(:,lastindex) = 2*pi;

        for i = 2:Nsamples,
            [~,lastindex] = min(D(lastindex,:));
            indices(i) = lastindex;
            D(:,lastindex) = 2*pi;
        end

        Vsortedsub = Vsub(:,indices);
        phiSortedSub(:,step) = phiSub(indices);
        thetaSortedSub(:,step) = thetaSub(indices);
        

        % Visualize mesh based on computed configuration of particles
        figure(1);plot3(Vsortedsub(1,:),Vsortedsub(2,:),Vsortedsub(3,:),'.k','MarkerSize',15)
        hold on
        plot3(Vsortedsub(1,:),Vsortedsub(2,:),Vsortedsub(3,:)); title(num2str(step))
        hold off
        drawnow;
        % delete the ones selected
        
        size(V,2)
        
        indices = gpuArray(zeros(Nsamples,1));
        
        %starting the next one with the pointed closest in the remaining
        %list
        if(step<steps)
            temp = real(acos(complex(Vsortedsub(:,end)'*V)));
            [~,farthest] = max(temp);
            indices(1)= farthest;
        end

end

phiSortedSub = phiSortedSub*180/pi;
thetaSortedSub = thetaSortedSub*180/pi;

