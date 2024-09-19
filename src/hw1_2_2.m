load('hall.mat');

hall_color(1:2:end, 1:2:end) = 0;
hall_color(2:2:end, 2:2:end) = 0;
imshow(hall_color);
exportgraphics(gcf, 'hw1_2_2.png', 'Resolution', 190);
