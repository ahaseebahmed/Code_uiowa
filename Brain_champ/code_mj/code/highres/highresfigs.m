
addpath('./../exportfig');

xrange1 = [50,120];
yrange1 = [50,150];
zrange1 = [90,145];
y1=25;
x1=50;
%%

load('/Volumes/lss_jcb/hemant/denoised/denoised_P76800_200_tv.mat')

mtx = 200;

xrange = xrange1*mtx/200;
yrange = yrange1*mtx/200;
zrange = zrange1*mtx/200;
xrange = xrange(1):xrange(2);
yrange = yrange(1):yrange(2);
zrange = zrange(1):zrange(2);
y11 = round(y1*mtx/200);
x11 = round(x1*mtx/200);

Volume2View = flip(denoised(xrange,yrange,zrange),3);
test = biasCorrect(Volume2View);
plotfigs(1.25*test,x1,y1,x11,y11,mtx,'denoised');


Volume2View = flip(noisy(xrange,yrange,zrange),3);
test = biasCorrect(1.5*Volume2View);
plotfigs(1.25*test,x1,y1,x11,y11,mtx,'noisy');

%% VolumeViewer3D((abs(Volume2View(:,:,:))))

load('/Volumes/lss_jcb/hemant/denoised/denoised_P76800_400_tv.mat')

mtx = 400;

xrange = floor(xrange1*mtx/200);
yrange = floor(yrange1*mtx/200);
zrange = floor(zrange1*mtx/200);

xrange = xrange(1):xrange(2);
yrange = yrange(1):yrange(2);
zrange = zrange(1):zrange(2);
y11 = round(y1*mtx/200);
x11 = round(x1*mtx/200);

Volume2View = flip(denoised(xrange,yrange,zrange),3);
test = biasCorrect(Volume2View);
plotfigs(test,x1,y1,x11,y11,mtx,'denoised');


Volume2View = flip(noisy(xrange,yrange,zrange),3);
test = biasCorrect(Volume2View);
plotfigs(test,x1,y1,x11,y11,mtx,'noisy');


%%

load('/Volumes/lss_jcb/hemant/denoised/denoised_P76800_700_tv.mat')

mtx = 700;

xrange = floor(xrange1*mtx/200);
yrange = floor(yrange1*mtx/200);
zrange = floor(zrange1*mtx/200);

xrange = xrange(1):xrange(2);
yrange = yrange(1):yrange(2);
zrange = zrange(1):zrange(2);
y11 = round(y1*mtx/200);
x11 = round(x1*mtx/200);

Volume2View = flip(denoised(xrange,yrange,zrange),3);
test = biasCorrect(Volume2View,2);
plotfigs(test,x1,y1,x11,y11,mtx,'denoised');


Volume2View = flip(noisy(xrange,yrange,zrange),3);
test = biasCorrect(Volume2View,2);
plotfigs(test,x1,y1,x11,y11,mtx,'noisy');

%%
%%

load('/Volumes/lss_jcb/hemant/denoised/denoised_P76800_600_5000_tv.mat')

mtx = 600;

xrange = floor(xrange1*mtx/200);
yrange = floor(yrange1*mtx/200);
zrange = floor(zrange1*mtx/200);

xrange = xrange(1):xrange(2);
yrange = yrange(1):yrange(2);
zrange = zrange(1):zrange(2);
y11 = round(y1*mtx/200);
x11 = round(x1*mtx/200);

Volume2View = flip(denoised(xrange,yrange,zrange),3);
test = biasCorrect(Volume2View);
plotfigs(test,x1,y1,x11,y11,mtx,'denoised');


Volume2View = flip(noisy(xrange,yrange,zrange),3);
test = biasCorrect(Volume2View);
plotfigs(test,x1,y1,x11,y11,mtx,'noisy');

%%
load('/Volumes/lss_jcb/abdul/P76800_700_700.mat')

mtx = 700;

xrange = floor(xrange1*mtx/200);
yrange = floor(yrange1*mtx/200);
zrange = floor(zrange1*mtx/200);

xrange = xrange(1):xrange(2);
yrange = yrange(1):yrange(2);
zrange = zrange(1):zrange(2);
y11 = round(y1*mtx/200);
x11 = round(x1*mtx/200);

Volume2View = flip(tvrecon(xrange,yrange,zrange),3);
test = biasCorrect(Volume2View);
plotfigs(test,x1,y1,x11,y11,mtx,'noisy');


% Volume2View = flip(noisy(xrange,yrange,zrange),3);
% test = biasCorrect(Volume2View);
% plotfigs(test,x1,y1,x11,y11,mtx,'noisy');
%%
load('/Shared/lss_jcb/hemant/denoised/denoised_P82432_400_5000.mat')

mtx = 400;

xrange = floor(xrange1*mtx/200);
yrange = floor(yrange1*mtx/200);
zrange = floor(zrange1*mtx/200);

xrange = xrange(1):xrange(2);
yrange = yrange(1):yrange(2);
zrange = zrange(1):zrange(2);
y11 = round(y1*mtx/200);
x11 = round(x1*mtx/200);

Volume2View = flip(denoised(xrange,yrange,zrange),3);
test = biasCorrect(Volume2View,2);
plotfigs(test,x1,y1,x11,y11,mtx,'denoised_P82432_');


Volume2View = flip(noisy(xrange,yrange,zrange),3);
test = biasCorrect(Volume2View,2);
plotfigs(test,x1,y1,x11,y11,mtx,'noisy_P82432_');