load('hall.mat');
load('jpegcodes.mat');
load('JpegCoeff.mat');

hall_jpeg = jpeg_decode(DC_code, AC_code, height, width, QTAB, DCTAB, ACTAB);

MSE = sum(sum((hall_gray - hall_jpeg) .^ 2)) / (height * width);
PSNR = 10 * log10(255 ^ 2 / MSE);

subplot(1, 2, 1);
imshow(hall_gray);
title('原图');
subplot(1, 2, 2);
imshow(hall_jpeg);
title('JPEG 编解码后的图像');
exportgraphics(gcf, 'hw2_11.png', 'Resolution', 190);
