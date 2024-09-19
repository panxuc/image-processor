load('hall.mat');

dct1 = dct2(hall_gray);
dct2 = my_dct2(hall_gray);

disp(norm(dct1 - dct2));