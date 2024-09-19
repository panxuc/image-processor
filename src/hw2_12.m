load('hall.mat');
load('JpegCoeff.mat');

[hall_jpeg_2, ~, ~, ~, ~, compress_ratio_2, PSNR_2] = jpeg_process(hall_gray, QTAB ./ 2, DCTAB, ACTAB);
[hall_jpeg, ~, ~, ~, ~, compress_ratio, PSNR] = jpeg_process(hall_gray, QTAB, DCTAB, ACTAB);

subplot(1, 3, 1);
imshow(hall_gray);
title('原图');
subplot(1, 3, 2);
imshow(hall_jpeg);
title('JPEG 编解码后的图像');
subplot(1, 3, 3);
imshow(hall_jpeg_2);
title('量化步长减小为原来的一半后 JPEG 编解码后的图像');
exportgraphics(gcf, 'hw2_12.png', 'Resolution', 190);
