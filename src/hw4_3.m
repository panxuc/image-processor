traindir = dir("Faces/*.bmp");
trainset = strings(length(traindir), 1);
for i = 1 : length(traindir)
    trainset(i) = strcat(traindir(i).folder, '/', traindir(i).name);
end

v5 = face_train(trainset, 5);

imag = imread('test_image.png');

imag_rotated = imrotate(imag, 90);
imag_resized = imresize(imag, [size(imag, 1), size(imag, 2) * 2]);
imag_colored = imadjust(imag, [0.2, 0.2, 0.2; 0.8, 0.8, 0.8]);

face = face_detect(imag, v5, 5, 0.7);
face_rotated = face_detect(imag_rotated, v5, 5, 0.7);
face_resized = face_detect(imag_resized, v5, 5, 0.7);
face_colored = face_detect(imag_colored, v5, 5, 0.7);

subplot(2, 2, 1);
imshow(face);
title('原图');
subplot(2, 2, 2);
imshow(face_rotated);
title('顺时针旋转 90°');
subplot(2, 2, 3);
imshow(face_resized);
title('保持高度不变，宽度拉伸为原来的 2 倍');
subplot(2, 2, 4);
imshow(face_colored);
title('适当改变颜色');
exportgraphics(gcf, 'hw4_3.png');
