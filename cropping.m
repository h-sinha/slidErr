function [rec] = cropping(norm_gradient)
%CROPPING Crops the image according to edgea
%   Will return non-cropped image if already cropped
colsum = sum(double(norm_gradient));
rowsum = sum(double(norm_gradient), 2);
[val1 ,idx1] = max(colsum(4:698));
[val2 ,idx2] = max(colsum(699:1394));
[val3 ,idx3] = max(rowsum(4:540));
[val4 ,idx4] = max(rowsum(540:1076));
if val1+val2+val3+val4 < 100000
    display("NOT Cropped")
    rec=[0 0 1440 1080];
else
    display("Cropped")
    rec=[idx1 idx3 idx2-idx1+699 idx4-idx3+540];
end

