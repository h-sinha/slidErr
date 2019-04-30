imgray = rgb2gray(imread('ppt.jpg'));
i1 = convert(imgray);
threshold = 10;
i1(i1>threshold) = 255;
i1(i1<=threshold) = 0;
imgray1 = rgb2gray(imread('0.jpg'));
i2 = convert(imgray1);
i2(i2>threshold) = 255;
i2(i2<=threshold) = 0;
max(max(normxcorr2(i1, i2)))

figure
subplot(1,2,1)
imshow(i1);
subplot(1,2,2)
imshow(i2);
function  norm_gradient = convert(imgray)
dx = conv2([-1 0 1; -2 0 2; -1 0 1], imgray);
dy = conv2([-1 -2 -1; 0 0 0; 1 2 1], imgray);
gradient = sqrt((dx.*dx) + (dy.*dy));
max_val = max(max(gradient));
min_val = min(min(gradient));
norm_gradient = uint8(round(255*(gradient-min_val)/(max_val-min_val)));
% colsum = sum(norm_gradient);
% rowsum = sum(norm_gradient,2);    
% [a,b] = max(colsum);
% max(max(normxcorr2(imgray, imgray1)))
% imshow(norm_gradient);
end
