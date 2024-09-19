load('snow.mat');
load('JpegCoeff.mat');

[snow_jpeg, ~, ~, ~, ~, compress_ratio, PSNR] = jpeg_process(snow, QTAB, DCTAB, ACTAB);

subplot(1, 2, 1);
imshow(snow);
title('原图');
subplot(1, 2, 2);
imshow(snow_jpeg);
title('JPEG 编解码后的图像');
exportgraphics(gcf, 'hw2_13.png', 'Resolution', 190);
