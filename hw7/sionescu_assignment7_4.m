
% Problem 7.4 - Lasso vs Ridge
% Silvia Ionescu
% 11-18-2016

clear; close all; clc;

% train and test datasets
quad_data = load('prostateStnd.mat');

xtrain = quad_data.Xtrain;
ytrain = quad_data.ytrain;

xtest = quad_data.Xtest;
ytest = quad_data.ytest;

mean = sum(xtrain,1)/67;

for i = 1:size(xtrain,2)
    xtrain(:,i) = xtrain(:,i) - mean(:,i);
end

% Part 4c - ridge regresion
reg = exp(-5:10);
for j = 1:length(reg)
    b_ridge_train(:,j) = ridge(ytrain, xtrain, reg(j), 0);
    
    w(:,j) = b_ridge_train(2:end,j);
    for t = 1:100  
        for k = 1:size(xtrain,2)
            a = 2*sum(xtrain(:,k).^2 );
            c = 2*xtrain(:,k)'*(ytrain - (w(:,j)'*xtrain')' + w(k,j).*xtrain(:,k));
            w(k,j) = sign(c/a)* max(0, abs(c/a)-(reg(j)/a));
        end
    end
end

figure(1)
plot(w(1,:));
hold on
plot(w(2,:));
hold on
plot(w(3,:));
hold on
plot(w(4,:));
hold on
plot(w(5,:));
hold on
plot(w(6,:));
hold on
plot(w(7,:));
hold on
plot(w(8,:));

legend('w1', 'w2', 'w3', 'w4', 'w5', 'w6', 'w7','w8');
ylabel('Lasso coefficients');
xlabel('ln(lambda)');
title('Lasso coefficients as a function of ln lambda');

% laasso coefficients - calculate predictions
for p =1:length(reg)
    y_pred_train = (w(:,p)'*xtrain')';
    y_pred_test = (w(:,p)'*xtest')';
    mse_train(p) = 1/length(ytrain)*sum((ytrain - y_pred_train).^2);
    mse_test(p) = 1/length(ytest)*sum((ytest - y_pred_test).^2);
    
    y_pred_train_ridge = (b_ridge_train(2:end,p)'*xtrain')';
    y_pred_test_ridge = (b_ridge_train(2:end,p)'*xtest')';
    mse_train_ridge(p) = 1/length(ytrain)*sum((ytrain - y_pred_train_ridge).^2);
    mse_test_ridge(p) = 1/length(ytest)*sum((ytest - y_pred_test_ridge).^2);
end

figure(2)
plot(mse_train)
hold on
plot(mse_test);
lgd = legend('mse train', 'mse test');
lgd.Location = 'southeast';
ylabel('MSE for training and testing')
xlabel('ln(lambda)');
title('Lasso MSE of both training and testing data as a function lo ln lambda ')
hold off

% ridge regression
figure(3)
plot(b_ridge_train(2,:));
hold on
plot(b_ridge_train(3,:));
hold on
plot(b_ridge_train(4,:));
hold on
plot(b_ridge_train(5,:));
hold on
plot(b_ridge_train(6,:));
hold on
plot(b_ridge_train(7,:));
hold on
plot(b_ridge_train(8,:));
hold on
plot(b_ridge_train(9,:));
legend('w1', 'w2', 'w3', 'w4', 'w5', 'w6', 'w7','w8');
ylabel('Ridge regression coefficients');
xlabel('ln(lambda)');
title('Ridge regression as a function of ln lambda');

figure(4)
plot(mse_train_ridge)
hold on
plot(mse_test_ridge);
lgd = legend('mse train', 'mse test');
lgd.Location = 'southeast';
ylabel('MSE for training and testing')
xlabel('ln(lambda)');
title('Ridge regression MSE of both training and testing data vs. ln ?. ')
hold off