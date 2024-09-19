load('hall.mat');
load('JpegCoeff.mat');

[DC_code, AC_code, height, width] = jpeg_encode(hall_gray, QTAB, DCTAB, ACTAB);

save jpegcodes.mat DC_code AC_code height width;
