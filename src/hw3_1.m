load('hall.mat');
load('JpegCoeff.mat');

info = 'Tsinghua University';

hall_hide = uint8(hide_spatial(hall_gray, info));
info_extract = extract_spatial(hall_hide)

[hall_hide_jpeg, ~, ~, ~, ~, ~, ~] = jpeg_process(hall_hide, QTAB, DCTAB, ACTAB);

info_extract_jpeg = extract_spatial(hall_hide_jpeg)

subplot(1, 3, 1);
imshow(hall_gray);
title('原图');
subplot(1, 3, 2);
imshow(hall_hide);
title('空域隐藏后的图像');
subplot(1, 3, 3);
imshow(hall_hide_jpeg);
title('空域隐藏、JPEG 编码后的图像');
exportgraphics(gcf, 'hw3_1.png', 'Resolution', 190);