im0 = imread('5.jpg');
imp0 = imread('ppt.jpg');
imgray = rgb2gray(im0);
norm_gradient = edgeDetection(imgray);
im0 = cropping(norm_gradient);
imp0 = edgeDetection(rgb2gray(imp0));
imp0 = cropping(imp0);
threshold = 10;
im0(im0>threshold) = 255;
im0(im0<=threshold) = 0;
imp0(imp0>threshold) = 255;
imp0(imp0<=threshold) = 0;
imp0 = imresize(imp0, size(im0));
max(max(normxcorr2(im0,imp0)))
figure;
subplot(1,2,1)
imshow(im0);
subplot(1,2,2)
imshow(imp0);

function norm_gradient = edgeDetection(imgray)
% Edge Detection
% Using Sobel Operator we can take the derivative of the image.
dx = conv2([-1 0 1; -2 0 2; -1 0 1], imgray);
dy = conv2([-1 -2 -1; 0 0 0; 1 2 1], imgray);
gradient = sqrt((dx.*dx) + (dy.*dy));
max_val = max(max(gradient));
min_val = min(min(gradient));
norm_gradient = uint8(round(255*(gradient-min_val)/(max_val-min_val)));
% imshow(norm_gradient);
end
function im0 = cropping(norm_gradient)
colsum = sum(double(norm_gradient));
rowsum = sum(double(norm_gradient), 2);
[val ,idx1] = max(colsum(4:698));
[val ,idx2] = max(colsum(699:1394));
[val ,idx3] = max(rowsum(4:540));
[val ,idx4] = max(rowsum(540:1076));
im0 = imcrop(norm_gradient, [idx1 idx3 idx2-idx1+699 idx4-idx3+540]);
end