function [Wt,bt] = interpolate_TLNs(W0,b0,W1,b1,t)

% function [Wt,bt] = interpolate_TLNs(W0,b0,W1,b1,t)
% function to interpolate between two TLNs (W0,b0) and (W1,b1) as:
% Wt = (1-t)*W0 + t*W1 and bt = (1-t)*b0 + t*b1
% written by Juliana L on oct 20 2023

n0 = size(W0,1);
n1 = size(W1,1);

if n0 ~= n1
    disp('TLN sizes are different!')
    return
else 
    n = n0;
end

% if nargin<3 || isempty(b0)
%     b0 = ones(n,1);
% end
% 
% if nargin<4 || isempty(b1)
%     b1 = ones(n,1);
% end

if nargin<5 || isempty(t)
    t = 0;
end

Wt = (1-t)*W0 + t*W1;
bt = (1-t)*b0 + t*b1;