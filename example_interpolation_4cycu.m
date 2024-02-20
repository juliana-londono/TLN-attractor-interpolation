% code to simulate one CTLN, and one TLN that has the same attractor
% the TLN was obtained using fit_TLN2TLN_feb22.py
% oct20: rounded W_final hoping to get a better match, added plot of
% difference between attractors

close all
clear all

T = 500;

% original CTLN (aka desired)

% this is a relabelled 4-cycu
sA_desired = [0,0,1,1;1,0,0,0;0,1,0,1;0,1,1,0];
sA_desired = sA_desired([3,4,1,2],[3,4,1,2]);
X0_desired = [0,0.1,0,0];
e = 0.25;
d = 0.5;
W_desired = graph2net(sA_desired,e,d);
b_desired = [1;1;1;1];
soln_desired = threshlin_ode(W_desired,b_desired,T,X0_desired);

% TLN with the same attractor

W_final = readmatrix('W_final.csv');
b_final = readmatrix('theta_final.csv');
X0_final = X0_desired;
soln_final = threshlin_ode(W_final,b_final,T,X0_final);

n = size(W_final,1);

figure(1)

%desired
subplot(4,9,[1,2,10,11])
imagesc(W_desired)
colormap(gray)
colorbar('Ticks',unique(W_desired),...
         'TickLabels',{'-1-\delta','-1+\epsilon','0'})
xticks(1:n)
yticks(1:n)
title('desired CTLN (4-cycu)')

subplot(4,9,[3,12])
imagesc(b_desired)
colormap(gray)
colorbar('Ticks',unique(b_desired))
xticks(1:n)
yticks(1:n)
title('theta')

subplot(4,9,4:8)
plot_grayscale(soln_desired.X);
title(['epsilon = ',num2str(e),', delta = ', num2str(d)])

subplot(4,9,13:17)
plot_ratecurves(soln_desired.X,soln_desired.time);

subplot(4,9,[9,18])
imagesc(X0_desired')
colormap(gray)
colorbar('Ticks',unique(X0_desired))
xticks(1:n)
yticks(1:n)
title('X0')

%final
subplot(4,9,[19,20,28,29])
imagesc(W_final)
colormap(gray)
colorbar('Ticks',unique(W_final))
xticks(1:n)
yticks(1:n)
title('final TLN')

subplot(4,9,[21,30])
imagesc(b_final)
colormap(gray)
colorbar('Ticks',unique(b_final))
xticks(1:n)
yticks(1:n)
title('b')

subplot(4,9,22:26)
plot_grayscale(soln_final.X);

subplot(4,9,31:35)
plot_ratecurves(soln_final.X,soln_final.time);

subplot(4,9,[27,36])
imagesc(X0_final')
colormap(gray)
colorbar('Ticks',unique(X0_final))
xticks(1:n)
yticks(1:n)
title('X0')

figure(2)
subplot(3,1,1)
plot_ratecurves(soln_desired.X,soln_desired.time);
title('desired')

subplot(3,1,2)
plot_ratecurves(soln_final.X,soln_final.time);
title('final')

subplot(3,1,3)
plot_ratecurves((soln_desired.X-soln_final.X),soln_final.time);
title('desired-final')





