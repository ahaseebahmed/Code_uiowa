function ktraj = giveGoldenAngleTraj(N,Nspokes,delta)

Nlow = -N/2+delta;
Nhigh = N/2-1+delta;

k = [Nlow:1:Nhigh];
ktraj = complex(zeros(N,Nspokes));

theta = 0;
goldenangle = pi*2/(1+sqrt(5));
for i=1:Nspokes,
    ktraj(:,i) = k*exp(1i*theta);
    theta = theta + goldenangle;
end
