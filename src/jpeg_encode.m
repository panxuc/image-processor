function [DC_code, AC_code, height, width] = jpeg_encode(imag, QTAB, DCTAB, ACTAB)
    block_size = 8;

    preprocessed = double(imag) - 128;
    [height, width] = size(preprocessed);
    
    height_blocks = ceil(height / block_size);
    width_blocks = ceil(width / block_size);
    
    result = [];
    for i = 1 : height_blocks
        for j = 1 : width_blocks
            block = preprocessed((i - 1) * block_size + 1 : i * block_size, (j - 1) * block_size + 1 : j * block_size);
            dct_block = dct2(block);
            quantized_block = dct_block ./ QTAB;
            zigzag_block = int32(zigzag_scan(quantized_block));
            result = [result, zigzag_block'];
        end
    end
    
    DC_components = double(result(1, :));
    DC_components = [DC_components(1), -diff(DC_components)];
    DC_code = [];
    category = dc_error_to_category(DC_components);
    for i = 1 : length(DC_components)
        if category(i) == 0
            DC_code = [DC_code, 0, 0];
        else
            huffman = DCTAB(category(i) + 1, 2 : DCTAB(category(i) + 1, 1) + 1);
            binary = dec2bin(abs(DC_components(i))) - '0';
            if (DC_components(i) < 0)
                binary = ~binary;
            end
            DC_code = [DC_code, huffman, binary];
        end
    end
    
    F0 = '11111111001' - '0';
    EOB = '1010' - '0';
    AC_code = [];
    for i = 1 : length(DC_components)
        AC_components = double(result(2 : end, i));
        AC_nonzero = find(AC_components);
        last_index = 0;
        while ~isempty(AC_nonzero)
            run = AC_nonzero(1) - last_index - 1;
            while run >= 16
                AC_code = [AC_code, F0];
                run = run - 16;
            end
            run_size = dc_error_to_category(AC_components(AC_nonzero(1)));
            huffman = ACTAB(run * 10 + run_size, 4 : ACTAB(run * 10 + run_size, 3) + 3);
            binary = dec2bin(abs(AC_components(AC_nonzero(1)))) - '0';
            if (AC_components(AC_nonzero(1)) < 0)
                binary = ~binary;
            end
            AC_code = [AC_code, huffman, binary];
            last_index = AC_nonzero(1);
            AC_nonzero = AC_nonzero(2 : end);
        end
        AC_code = [AC_code, EOB];
    end
end
