dc_error = -2047:2047;
category = dc_error_to_category(dc_error);
plot(dc_error, category);
xlabel('DC 预测误差');
ylabel('Category 值');
title('DC 预测误差与 Category 值的关系');
exportgraphics(gcf, 'hw2_6.png');
