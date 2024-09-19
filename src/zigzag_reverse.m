function matrix = zigzag_reverse(zigzag)
    len = length(zigzag);
    n = ceil(sqrt(len));
    matrix = zeros(n, n);
    i = 1;
    for s = 1:2*n-1
        if mod(s, 2) == 1
            for y = 1:min(s, n)
                x = s - y + 1;
                if x >= 1 && x <= n && y >= 1 && y <= n
                    matrix(x, y) = zigzag(i);
                    i = i + 1;
                end
            end
        else
            for x = 1:min(s, n)
                y = s - x + 1;
                if x >= 1 && x <= n && y >= 1 && y <= n
                    matrix(x, y) = zigzag(i);
                    i = i + 1;
                end
            end
        end
    end
end