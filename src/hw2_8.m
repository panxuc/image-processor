load('hall.mat');
load('JpegCoeff.mat');

block_size = 8;

hall_preprocessed = double(hall_gray) - 128;
[height, width] = size(hall_preprocessed);

height_blocks = ceil(height / block_size);
width_blocks = ceil(width / block_size);

height_extended = height_blocks * block_size;
width_extended = width_blocks * block_size;

result = [];
for i = 1 : height_blocks
    for j = 1 : width_blocks
        block = hall_preprocessed((i - 1) * block_size + 1 : i * block_size, (j - 1) * block_size + 1 : j * block_size);
        dct_block = dct2(block);
        quantized_block = dct_block ./ QTAB;
        zigzag_block = int32(zigzag_scan(quantized_block));
        result = [result, zigzag_block'];
    end
end
