function movie_interpolate_TLNs(W0,b0,W1,b1,I,T,f)
% function movie_interpolate_TLNs(W0,b0,W1,b1,I,T,f)
% movie to interpolate between two TLNs (W0,b0) and (W1,b1) for all t in
% the inverval I. Simulation will run for a duration of T time units. f is
% the number of frames to generate. default is 10.
% output is just the movie

% things TO DO: plot the difference matrix, change the filename that the movie saves to!! 

n0 = size(W0,1);
n1 = size(W1,1);

if n0 ~= n1
    disp('TLN sizes are different!')
    return
else 
    n = n0;
end

if nargin<5 || isempty(I)
    I = [0,1];
end

if nargin<6 || isempty(T)
    T = 200;
end

if nargin<7 || isempty(f)
    f = 10;
end

X0 = 0.1*ones(n,1);
X0(1) = 0.2;

%Set up the axes and figure properties to generate frames for the video.
fh = figure();
fh.WindowState = 'maximized';
axis tight manual
set(gca,"NextPlot","replacechildren")

%Create a VideoWriter object for the output video file and open the object for writing.
filename = sprintf('example_interpolation_movie_%s', string(datetime,'MMM_dd_HHmmss'));
v = VideoWriter(filename,'MPEG-4');
v.FrameRate = 2; %default is 30
open(v)

a = I(1);
b = I(2);

for t = linspace(a,b,f)
    [Wt,bt] = interpolate_TLNs(W0,b0,W1,b1,t);
    %solve
    solnt = threshlin_ode(Wt,bt,T,X0);
    %compute FP
    subplot(2,9,[1,2,10,11])
    imagesc(Wt)
    colormap(gray)
    colorbar('Ticks',unique(Wt))
    xticks(1:n)
    yticks(1:n)
    title('Wt')

    subplot(2,9,[3,12])
    imagesc(bt)
    colormap(gray)
    colorbar('Ticks',unique(bt))
    xticks(1:n)
    yticks(1:n)
    title('bt')

    subplot(2,9,4:8)
    plot_grayscale(solnt.X);
    title(['t = ',num2str(t)])

    subplot(2,9,13:17)
    plot_ratecurves(solnt.X,solnt.time);
    %title(['FP(Wt,bt) = '])

    subplot(2,9,[9,18])
    imagesc(X0)
    colormap(gray)
    colorbar('Ticks',unique(X0))
    xticks(1:n)
    yticks(1:n)
    title('X0')

    frame = getframe(gcf);
    writeVideo(v,frame)
end
close(v)
end

