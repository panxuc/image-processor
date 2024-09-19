load('hall.mat');

dct_hall_transpose = dct2(hall_gray);
dct_hall_transpose = dct_hall_transpose';
hall_gray_transpose = idct2(dct_hall_transpose);

dct_hall_90 = dct2(hall_gray);
dct_hall_90 = rot90(dct_hall_90);
hall_gray_90 = idct2(dct_hall_90);

dct_hall_180 = dct2(hall_gray);
dct_hall_180 = rot90(dct_hall_180, 2);
hall_gray_180 = idct2(dct_hall_180);

subplot(1, 4, 1);
imshow(hall_gray);
title('原图');
subplot(1, 4, 2);
imshow(uint8(hall_gray_transpose));
title('对 DCT 系数转置');
subplot(1, 4, 3);
imshow(uint8(hall_gray_90));
title('对 DCT 系数旋转 90 度');
subplot(1, 4, 4);
imshow(uint8(hall_gray_180));
title('对 DCT 系数旋转 180 度');
exportgraphics(gcf, 'hw2_4.png', 'Resolution', 190);