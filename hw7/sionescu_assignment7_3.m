% Problem 7.3 - Overfitting and ridge regression
% Silvia Ionescu
% 11-18-2016

clear; close all; clc;

% train and test datasets
quad_data = load('quad_data.mat');
xtrain_a = quad_data.xtrain;
ytrain = quad_data.ytrain;

xtest_a = quad_data.xtest;
ytest = quad_data.ytest;

for i = 1:14
    xtrain(:,i) = xtrain_a.^i; 
    xtest(:,i) = xtest_a.^i;

end

% Part 7_3a
% calculating coefficients for 1...14 degrees for training data
b1 = ridge(ytrain, xtrain(:,1), 0, 0);
b2 = ridge(ytrain, xtrain(:,1:2), 0, 0);
b3 = ridge(ytrain, xtrain(:,1:3), 0, 0);
b4 = ridge(ytrain, xtrain(:,1:4), 0, 0);
b5 = ridge(ytrain, xtrain(:,1:5), 0, 0);
b6 = ridge(ytrain, xtrain(:,1:6), 0, 0);
b7 = ridge(ytrain, xtrain(:,1:7), 0, 0);
b8 = ridge(ytrain, xtrain(:,1:8), 0, 0);
b9 = ridge(ytrain, xtrain(:,1:9), 0, 0);
b10 = ridge(ytrain, xtrain(:,1:10), 0, 0);
b11 = ridge(ytrain, xtrain(:,1:11), 0, 0);
b12 = ridge(ytrain, xtrain(:,1:12), 0, 0);
b13 = ridge(ytrain, xtrain(:,1:13), 0, 0);
b14 = ridge(ytrain, xtrain(:,1:14), 0, 0);

y1 = b1(2)'*xtrain(:,1)' + b1(1);
y2 = b2(2:end)'*xtrain(:,1:2)' + b2(1);
y3 = b3(2:end)'*xtrain(:,1:3)' + b3(1);
y4 = b4(2:end)'*xtrain(:,1:4)' + b4(1);
y5 = b5(2:end)'*xtrain(:,1:5)' + b5(1);
y6 = b6(2:end)'*xtrain(:,1:6)' + b6(1);
y7 = b7(2:end)'*xtrain(:,1:7)' + b7(1);
y8 = b8(2:end)'*xtrain(:,1:8)' + b8(1);
y9 = b9(2:end)'*xtrain(:,1:9)' + b9(1);
y10 = b10(2:end)'*xtrain(:,1:10)' + b10(1);
y11 = b11(2:end)'*xtrain(:,1:11)' + b11(1);
y12 = b12(2:end)'*xtrain(:,1:12)' + b12(1);
y13 = b13(2:end)'*xtrain(:,1:13)' + b13(1);
y14 = b14(2:end)'*xtrain(:,1:14)' + b14(1);

y_train_predict = [y1;y2;y3;y4;y5;y6;y7;y8;y9;y10;y11;y12;y13;y14]';

y1_t = b1(2)'*xtest(:,1)' + b1(1);
y2_t = b2(2:end)'*xtest(:,1:2)' + b2(1);
y3_t = b3(2:end)'*xtest(:,1:3)' + b3(1);
y4_t = b4(2:end)'*xtest(:,1:4)' + b4(1);
y5_t = b5(2:end)'*xtest(:,1:5)' + b5(1);
y6_t = b6(2:end)'*xtest(:,1:6)' + b6(1);
y7_t = b7(2:end)'*xtest(:,1:7)' + b7(1);
y8_t = b8(2:end)'*xtest(:,1:8)' + b8(1);
y9_t = b9(2:end)'*xtest(:,1:9)' + b9(1);
y10_t = b10(2:end)'*xtest(:,1:10)' + b10(1);
y11_t = b11(2:end)'*xtest(:,1:11)' + b11(1);
y12_t = b12(2:end)'*xtest(:,1:12)' + b12(1);
y13_t = b13(2:end)'*xtest(:,1:13)' + b13(1);
y14_t = b14(2:end)'*xtest(:,1:14)' + b14(1);

y_test_predict = [y1_t; y2_t; y3_t; y4_t; y5_t; y6_t; y7_t; y8_t; y9_t; y10_t; y11_t; y12_t; y13_t; y14_t]';

% training point and polynomial curves of degrees 2,6,10,14
figure(1)
scatter(xtrain(:,1),ytrain);
hold on
plot(y2);
hold on
plot(y6);
hold on
plot(y10)
hold on
plot(y14)
lgd = legend ('poly degree2', 'poly degree6', 'poly degree10', 'poly degree14');
lgd.Location = 'southeast';
title('Training points with polynomial curves of degree 2, 6, 10, 14');
ylabel('Ytrain');
xlabel('Xtrain')
hold off

% calculate mse for train data
for j = 1:14
    mse_train(j) = 1/length(ytrain)*sum((ytrain-y_train_predict(:,j)).^2);
    mse_test(j) = 1/length(ytest)*sum((ytest-y_test_predict(:,j)).^2);
end

% mean-squared error as a function of polynomial degree

figure(2)
plot(1:14, mse_train);
hold on
plot(1:14, mse_test);
legend('mse train', 'mse test');
xlabel('Polynomial degree');
ylabel('MSE test and train');
title ('Training and testing data MSE vs. polynomial degree d= 1,...14');
hold off;

% Part b
reg = exp(-25:5);
for k = 1:length(reg)
    b10_ridge_train = ridge(ytrain, xtrain(:,1:10), reg(k), 0);
    y10_ridge_train = b10_ridge_train(2:end)'*xtrain(:,1:10)' + b10_ridge_train(1);
    mse_train_ridge(k) = 1/length(ytrain)*sum((ytrain-y10_ridge_train').^2);
    
    y10_ridge_test(:,k) = (b10_ridge_train(2:end)'*xtest(:,1:10)' + b10_ridge_train(1))';
    mse_test_ridge(k) = 1/length(ytest)*sum((ytest-y10_ridge_test(:,k)).^2);
    
    b4_ridge_train(:,k) = ridge(ytrain, xtrain(:,1:4), reg(k), 0);
    
end
figure(3)
plot(mse_train_ridge);
hold on
plot(mse_test_ridge);
hold off
lgd = legend('mse train', 'mse test');
lgd.Location = 'southeast';
xlabel('ln(lambda)');
ylabel('MSE traing and testing data');
title('MSE of the training and testing data vs.ln ?');

min_test_mse = find(mse_test_ridge == min(mse_test_ridge));
% Part 3bii
figure(4)
scatter(xtest(:,1),ytest);
hold on
plot(xtest(:,1),y10_t)
hold on
plot(xtest(:,1),y10_ridge_test(:,min_test_mse));
xlabel('xtest data');
ylabel('Polynomial fit'); 
lgd = legend('data','OLS', 'l2');
lgd.Location = 'southeast';
title('non-regularized OLS degree 10 polynomial fit and the l2-regularized degree 10 fit');

% Part 3c
figure(5)
plot(b4_ridge_train(1,:))
hold on
plot(b4_ridge_train(2,:))
hold on
plot(b4_ridge_train(3,:))
hold on
plot(b4_ridge_train(4,:))
hold on
plot(b4_ridge_train(5,:))
legend('w0', 'w1', 'w2', 'w3', 'w5');
xlabel('ln(lamda)');
ylabel('Coefficient values');
title('Coefficient values vs. ln(lamda)');
