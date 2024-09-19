load('hall.mat');

dct_hall_right = dct2(hall_gray);
dct_hall_right(:, end - 3:end) = 0;
hall_gray_right = idct2(dct_hall_right);

dct_hall_left = dct2(hall_gray);
dct_hall_left(:, 1:4) = 0;
hall_gray_left = idct2(dct_hall_left);

subplot(1, 3, 1);
imshow(hall_gray);
title('原图');
subplot(1, 3, 2);
imshow(uint8(hall_gray_right));
title('右侧四列的系数全部置零');
subplot(1, 3, 3);
imshow(uint8(hall_gray_left));
title('左侧四列的系数全部置零');
exportgraphics(gcf, 'hw2_3.png', 'Resolution', 190);
