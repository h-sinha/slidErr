function [norm_gradient] = edgedetect(imgray)
%EDGEDETECT Returns edge detected image
%   Using Sobel Operator we can take the derivative of the image.
dx = conv2([-1 0 1; -2 0 2; -1 0 1], imgray);
dy = conv2([-1 -2 -1; 0 0 0; 1 2 1], imgray);
gradient = sqrt((dx.*dx) + (dy.*dy));
max_val = max(max(gradient));
min_val = min(min(gradient));
norm_gradient = uint8(round(255*(gradient-min_val)/(max_val-min_val)));
end

