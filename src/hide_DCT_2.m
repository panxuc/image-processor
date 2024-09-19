function [jpeg, info_extracted, DC_code, AC_code, height, width, compress_ratio, PSNR] = hide_DCT_2(imag, info, QTAB, DCTAB, ACTAB)
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

    result_clip = result(1 : floor(end / 4), :);
    result_clip = hide_spatial(result_clip, info);
    result_clip(result_clip > 127) = result_clip(result_clip > 127) - 256;
    result(1 : floor(end / 4), :) = result_clip;
    
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

    compress_ratio = (height * width * 8) / (length(DC_code) + length(AC_code));

    block_num = height_blocks * width_blocks;
    
    decoded = zeros(64, block_num);
    
    index = 1;
    [max_category, ~] = size(DCTAB);
    step = DCTAB(:, 1);
    for i = 1 : block_num
        for category = 1 : max_category
            if (DC_code(index : index + step(category) - 1) == DCTAB(category, 2 : step(category) + 1))
                break;
            end
        end
        index = index + step(category);
        binary = DC_code(index : index + category - 2);
        index = index + category - 1;
        if (category > 1)
            if (binary(1) == 0)
                binary = ~binary;
                decoded(1, i) = -binary * 2 .^ (category - 2 : -1 : 0)';
            else
                decoded(1, i) = binary * 2 .^ (category - 2 : -1 : 0)';
            end
        else
            decoded(1, i) = 0;
        end
        if (i > 1)
            decoded(1, i) = decoded(1, i - 1) - decoded(1, i);
        end
    end
    
    index = 1;
    [max_run_size, ~] = size(ACTAB);
    AC_code = [AC_code, zeros(1, 16)];
    for i = 1 : block_num
        sum_run = 1;
        while (sum_run < 64)
            if (AC_code(index : index + 3) == EOB)
                index = index + 4;
                break;
            end
            if (AC_code(index : index + 10) == F0)
                sum_run = sum_run + 16;
                index = index + 11;
                continue;
            end
            for run_size = 1 : max_run_size
                step = ACTAB(run_size, 3);
                if (ACTAB(run_size, 4 : step + 3) == AC_code(index : index + step - 1))
                    break;
                end
            end
            index = index + step;
            run = ACTAB(run_size, 1);
            binary_size = ACTAB(run_size, 2);
            binary = AC_code(index : index + binary_size - 1);
            index = index + binary_size;
            sum_run = sum_run + run + 1;
            if (binary(1) == 0)
                binary = ~binary;
                decoded(sum_run, i) = -binary * 2 .^ (binary_size - 1 : -1 : 0)';
            else
                decoded(sum_run, i) = binary * 2 .^ (binary_size - 1 : -1 : 0)';
            end
        end
    end

    decoded_clip = decoded(1 : floor(end / 4), :);
    info_extracted = extract_spatial(decoded_clip);
    
    jpeg = zeros(height_blocks * 8, width_blocks * 8);
    
    index = 1;
    for i = 1 : 8 : height_blocks * 8
        for j = 1 : 8 : width_blocks * 8
            jpeg(i : i + 7, j : j + 7) = idct2(zigzag_reverse(decoded(:, index)) .* QTAB);
            index = index + 1;
        end
    end
    
    jpeg = uint8(jpeg(1 : height, 1 : width) + 128);

    MSE = sum(sum((imag - jpeg) .^ 2)) / (height * width);
    PSNR = 10 * log10(255 ^ 2 / MSE);
end