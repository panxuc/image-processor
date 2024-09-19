traindir = dir("Faces/*.bmp");
trainset = strings(length(traindir), 1);
for i = 1 : length(traindir)
    trainset(i) = strcat(traindir(i).folder, '/', traindir(i).name);
end

v3 = face_train(trainset, 3);
v4 = face_train(trainset, 4);
v5 = face_train(trainset, 5);

subplot(3, 1, 1);
plot(v3);
title('L = 3');
subplot(3, 1, 2);
plot(v4);
title('L = 4');
subplot(3, 1, 3);
plot(v5);
title('L = 5');
exportgraphics(gcf, 'hw4_1.png', 'Resolution', 190);
