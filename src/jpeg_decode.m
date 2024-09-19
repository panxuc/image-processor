function jpeg = jpeg_decode(DC_code, AC_code, height, width, QTAB, DCTAB, ACTAB)
    block_size = 8;
      
    height_blocks = ceil(height / block_size);
    width_blocks = ceil(width / block_size);
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
    
    F0 = '11111111001' - '0';
    EOB = '1010' - '0';
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
    
    jpeg = zeros(height_blocks * 8, width_blocks * 8);
    
    index = 1;
    for i = 1 : 8 : height_blocks * 8
        for j = 1 : 8 : width_blocks * 8
            jpeg(i : i + 7, j : j + 7) = idct2(zigzag_reverse(decoded(:, index)) .* QTAB);
            index = index + 1;
        end
    end
    
    jpeg = uint8(jpeg(1 : height, 1 : width) + 128);
end
