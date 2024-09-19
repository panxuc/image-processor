load('hall.mat');

[height, width, ~] = size(hall_color);
center = [width + 1, height + 1] / 2;
radius = min(height, width) / 2;
imshow(hall_color);
viscircles(center, radius, 'Color', 'r');
exportgraphics(gcf, 'hw1_2_1.png', 'Resolution', 190);
