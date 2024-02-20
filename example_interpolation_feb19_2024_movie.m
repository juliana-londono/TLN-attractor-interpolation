% code to interpolate between a W0,b0 (4-cycu) and W1,b1 (a TLN that learnt the
% 4-cycu attractor). Finds FPs, plots attractor, makes movie.
% written by Juliana on oct 20 2023
% feb 19: updated () matrix that fits up to T = 500

close all
clear all


% W0,b0:
% this is a relabelled 4-cycu
sA0 = [0,0,1,1;1,0,0,0;0,1,0,1;0,1,1,0];
sA0 = sA0([3,4,1,2],[3,4,1,2]);
e0 = 0.25;
d0 = 0.5;
W0 = graph2net(sA0,e0,d0);
n = size(W0,1);
b0 = ones(n,1);

% W1,b1:
W1 = readmatrix('W_final.csv');
b1 = readmatrix('theta_final.csv');

I=[0,1]; %interpolation interval
T=200; %duration of simulation
f=100; %total frames

movie_interpolate_TLNs(W0,b0,W1,b1,I,T,f)
