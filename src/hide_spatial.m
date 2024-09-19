function imag_hide = hide_spatial(imag, info)
    bits = reshape(dec2bin(info, 8)', 1, []) - '0';
    imag_size = numel(imag);
    len = length(bits);
    if (imag_size <= len + 16)
        info_hided = [dec2bin(imag_size - 16, 16) - '0', bits(1 : imag_size - 16)];
    else
        info_hided = [dec2bin(len, 16) - '0', bits];
    end
    cipher = dec2bin(reshape(imag', 1, []), 8);
    cipher(1 : len + 16, 8) = info_hided + '0';
    imag_hide = reshape(bin2dec(cipher), size(imag'))';
end
