load('hall.mat');

dct_hall = dct2(hall_gray);
dct_hall(1, 1) = dct_hall(1, 1) - 128 * sqrt(numel(dct_hall));
hall_gray_ = idct2(dct_hall);

diff = double(hall_gray) - 128 - hall_gray_;
disp(norm(diff));

figure;
subplot(1, 3, 1);
imshow(hall_gray);
title('原图');
subplot(1, 3, 2);
imshow(uint8(hall_gray_));
title('变换域处理结果');
subplot(1, 3, 3);
imshow(uint8(hall_gray - 128));
title('直接处理结果');
exportgraphics(gcf, 'hw2_1.png', 'Resolution', 190);
