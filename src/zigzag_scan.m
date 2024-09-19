function zigzag = zigzag_scan(matrix)
    [height, width] = size(matrix);
    zigzag = zeros(1, height * width);
    i = 1;
    for s = 1:height + width - 1
        if mod(s, 2) == 1
            for y = 1:min(s, width)
                x = s - y + 1;
                if x <= height && y <= width
                    zigzag(i) = matrix(x, y);
                    i = i + 1;
                end
            end
        else
            for x = 1:min(s, height)
                y = s - x + 1;
                if x <= height && y <= width
                    zigzag(i) = matrix(x, y);
                    i = i + 1;
                end
            end
        end
    end
end
