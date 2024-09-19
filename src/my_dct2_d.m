function D = my_dct2_d(N)
    D = zeros(N, N);
    for i = 1:N
        for j = 1:N
            D(i, j) = cos((i - 1) * (j * 2 - 1) * pi / (N * 2));
        end
    end
    D(1, :) = D(1, :) * sqrt(1 / 2);
    D = D * sqrt(2 / N);
end
