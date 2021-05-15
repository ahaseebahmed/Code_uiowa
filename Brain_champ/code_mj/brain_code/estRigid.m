function [tPhysical,t,reg_img] = estRigid(moving,fixed)

moving=moving./max(moving(:));
moving=imadjustn(moving);
fixed=fixed./max(fixed(:));
fixed=imadjustn(fixed);

[optimizer,metric] = imregconfig('monomodal');
%optimizer = registration.optimizer.OnePlusOneEvolutionary;
%optimizer.InitialRadius = 0.004;

% optimizer.Epsilon = 1.5e-4;
% optimizer.GrowthFactor = 1.01;
optimizer.MaximumIterations = 300;
%optimizer.GradientMagnitudeTolerance=1e-10;
%metric = registration.metric.MeanSquares;
% Rfixed  = imref3d(size(fixed),1,1,1);
% Rmoving = imref3d(size(moving),1,1,1);
%t0 = imregtform(abs(moving), abs(fixed), 'rigid', optimizer, metric);
t = imregtform(abs(moving), abs(fixed), 'rigid', optimizer, metric);
reg_img = imregister(abs(moving), abs(fixed), 'rigid', optimizer, metric,'DisplayOptimization',true);

R = imref3d(size(moving));
center = ([mean(R.XWorldLimits),mean(R.YWorldLimits),mean(R.ZWorldLimits)]);
%[xWorld,yWorld,zWorld] = transformPointsForward(t,center(1),center(2),center(3));


tPhysical = t;
translation = eye(4);translation(4,1:3)= center;
tPhysical.T = translation*(t.T)*inv(translation);

end

