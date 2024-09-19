traindir = dir("Faces/*.bmp");
trainset = strings(length(traindir), 1);
for i = 1 : length(traindir)
    trainset(i) = strcat(traindir(i).folder, '/', traindir(i).name);
end

v3 = face_train(trainset, 3);
v4 = face_train(trainset, 4);
v5 = face_train(trainset, 5);

imag = imread('test_image.png');

face3 = face_detect(imag, v3, 3, 0.4);
face4 = face_detect(imag, v4, 4, 0.55);
face5 = face_detect(imag, v5, 5, 0.7);

subplot(1, 3, 1);
imshow(face3);
title('L = 3');
subplot(1, 3, 2);
imshow(face4);
title('L = 4');
subplot(1, 3, 3);
imshow(face5);
title('L = 5');
exportgraphics(gcf, 'hw4_2.png');
