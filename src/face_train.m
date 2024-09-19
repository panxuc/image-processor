function output = face_train(trainset, L)
    if (L < 1 || L > 8 || floor(L) ~= L)
        error('L is illegal');
    end
    output = zeros(1, 2 ^ (3 * L));
    for i = 1 : size(trainset, 1)
        image = imread(trainset(i));
        u = zeros(1, 2 ^ (3 * L));
        R = dec2bin(reshape(image(:, :, 1), 1, []), 8) - '0';
        G = dec2bin(reshape(image(:, :, 2), 1, []), 8) - '0';
        B = dec2bin(reshape(image(:, :, 3), 1, []), 8) - '0';
        c_n = R(:, 1 : L) * 2 .^ (3 * L - 1 : -1 : 2 * L)' + G(:, 1 : L) * 2 .^ (2 * L - 1 : -1 : L)' + B(:, 1 : L) * 2 .^ (L - 1 : -1 : 0)' + 1;
        for n = 1 : length(c_n)
            u(c_n(n)) = u(c_n(n)) + 1;
        end
        u = u / numel(c_n);
        output = output + u;
    end
    output = output / size(trainset, 1);
end
