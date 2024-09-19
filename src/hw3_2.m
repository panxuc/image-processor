load('hall.mat');
load('JpegCoeff.mat');

info = 'Tsinghua University';

[hall_hide_0, ~, ~, ~, ~, compress_ratio_0, PSNR_0] = jpeg_process(hall_gray, QTAB, DCTAB, ACTAB);
[hall_hide_1, info_1, ~, ~, ~, ~, compress_ratio_1, PSNR_1] = hide_DCT_1(hall_gray, info, QTAB, DCTAB, ACTAB);
[hall_hide_2, info_2, ~, ~, ~, ~, compress_ratio_2, PSNR_2] = hide_DCT_2(hall_gray, info, QTAB, DCTAB, ACTAB);
[hall_hide_3, info_3, ~, ~, ~, ~, compress_ratio_3, PSNR_3] = hide_DCT_3(hall_gray, info, QTAB, DCTAB, ACTAB);

subplot(2, 2, 1);
imshow(hall_hide_0);
title('直接 JPEG 编解码的图像');
subplot(2, 2, 2);
imshow(hall_hide_1);
title('方法 1 隐藏后 JPEG 编解码的图像');
subplot(2, 2, 3);
imshow(hall_hide_2);
title('方法 2 隐藏后 JPEG 编解码的图像');
subplot(2, 2, 4);
imshow(hall_hide_3);
title('方法 3 隐藏后 JPEG 编解码的图像');
exportgraphics(gcf, 'hw3_2.png', 'Resolution', 190);
