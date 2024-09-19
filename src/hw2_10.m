load('hall.mat');
load('JpegCoeff.mat');

[DC_code, AC_code, height, width] = jpeg_encode(hall_gray, QTAB, DCTAB, ACTAB);

compress_ratio = (height * width * 8) / (length(DC_code) + length(AC_code));
disp(compress_ratio);
