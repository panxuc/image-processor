function [jpeg, DC_code, AC_code, height, width, compress_ratio, PSNR] = jpeg_process(imag, QTAB, DCTAB, ACTAB)
    [DC_code, AC_code, height, width] = jpeg_encode(imag, QTAB, DCTAB, ACTAB);
    
    compress_ratio = (height * width * 8) / (length(DC_code) + length(AC_code));
    
    jpeg = jpeg_decode(DC_code, AC_code, height, width, QTAB, DCTAB, ACTAB);
    
    MSE = sum(sum((imag - jpeg) .^ 2)) / (height * width);
    PSNR = 10 * log10(255 ^ 2 / MSE);
end
