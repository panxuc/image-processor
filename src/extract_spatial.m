function info = extract_spatial(imag)
    cipher = dec2bin(reshape(imag', 1, []), 8);
    info_hided = cipher(:, 8);
    len = bin2dec(char(info_hided(1 : 16)'));
    imag_size = numel(imag);
    if (imag_size <= len + 16)
        bits = info_hided(17 : imag_size + 16);
    else
        bits = info_hided(17 : len + 16);
    end
    try
        info = char(bin2dec(reshape(bits, 8, [])'))';
    catch
        info = '**Unable to extract info**';
    end
end
