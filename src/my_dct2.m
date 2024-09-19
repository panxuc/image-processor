function dct = my_dct2(p)
    [height, width] = size(p);
    dct = my_dct2_d(height) * double(p) * my_dct2_d(width)';
end
