%% Import & Crop Images

captureRGB = imread('08_2/0.jpg');
slide_original = imread('08_2/ppt.jpg');
capture_original = rgb2gray(captureRGB);
edges = cropping(edgedetect(capture_original));
capture = imcrop(capture_original, edges);
slide = imresize(rgb2gray(slide_original), size(capture));
subplot(1,2,1)
imshow(capture);
subplot(1,2,2)
imshow(slide);
figure;
%% Correlation
%%