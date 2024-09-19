function imag_marked = face_detect(imag, v, L, precision)
    if (L < 1 || L > 8 || floor(L) ~= L)
        error('L is illegal');
    end
    [height, width, ~] = size(imag);
    block_size = 32;
    height_blocks = ceil(height / block_size);
    width_blocks = ceil(width / block_size);
    block_is_face = false([height_blocks, width_blocks]);
    faces = [];

    for i = 1 : height_blocks
        for j = 1 : width_blocks
            block = imag((i - 1) * block_size + 1 : min(i * block_size, height), (j - 1) * block_size + 1 : min(j * block_size, width), :);
            block_v = zeros(1, 2 ^ (3 * L));
            R = dec2bin(reshape(block(:, :, 1), 1, []), 8) - '0';
            G = dec2bin(reshape(block(:, :, 2), 1, []), 8) - '0';
            B = dec2bin(reshape(block(:, :, 3), 1, []), 8) - '0';
            c_n = R(:, 1 : L) * 2 .^ (3 * L - 1 : -1 : 2 * L)' + G(:, 1 : L) * 2 .^ (2 * L - 1 : -1 : L)' + B(:, 1 : L) * 2 .^ (L - 1 : -1 : 0)' + 1;
            for n = 1 : length(c_n)
                block_v(c_n(n)) = block_v(c_n(n)) + 1;
            end
            block_v = block_v / numel(c_n);
            distance = 1 - sum(sqrt(block_v .* v), 'all');
            block_is_face(i, j) = distance < precision;
        end
    end

    block_is_face = bwareaopen(block_is_face, 9);

    label = bwlabel(block_is_face);
    num = max(label, [], 'all');
    imag_marked = imag;
    if num > 0
        for index = 1 : num
            [x, y] = find(label == index);
            x1 = min(x); x2 = max(x);
            y1 = min(y); y2 = max(y);
            faces = [faces; x1, y1, x2, y2];
            imag_marked = insertShape(imag_marked, 'Rectangle', [(y1 - 1) * block_size + 1, (x1 - 1) * block_size + 1, (y2 - y1 + 1) * block_size, (x2 - x1 + 1) * block_size], 'Color', 'red', 'LineWidth', 8);
        end
    end
end
