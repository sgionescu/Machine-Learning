

% Problem 7.2 ? Ordinary Least Squares (OLS) vs. Robust Linear Regression
% Silvia Ionescu
% 11-18-2016

clear; close all; clc;

% train and test datasets
housing_data = load('linear_data.mat');
xData = housing_data.xData;
yData = housing_data.yData;

ux = sum(xData)/length(xData);
uy = sum(yData)/length(yData);

cov_x = 1/length(xData)*((xData - ux)'*(xData - ux));
cov_xy = 1/length(xData)*sum((yData - uy).*(xData - ux));

w_ols = (cov_x)^-1 * cov_xy;
b_ols = uy - w_ols * ux;

xData_ux = xData - ux; 
yData_uy = yData - uy;

%w_ols2 = (1/length(xData)*(xData_ux'*xData_ux))^-1 *1/length(xData)*(xData_ux'*yData_uy);

% calculate MSE - mean squared error
h_ols = xData*w_ols + b_ols;
mse_ols = 1/length(xData)*sum((yData-h_ols).^2);
mae_ols = 1/length(xData)*sum(abs(yData-h_ols));

% Part b
b_c = robustfit(xData, yData, 'cauchy'); 
b_f = robustfit(xData, yData, 'fair'); 
b_h = robustfit(xData, yData, 'huber'); 
b_t = robustfit(xData, yData, 'talwar');

% predicted y 
y_pred_cauchy = b_c(1)+b_c(2)*xData;
y_pred_fair = b_f(1)+b_f(2)*xData;
y_pred_huber = b_h(1)+b_h(2)*xData;
y_pred_talwar = b_t(1)+b_t(2)*xData;

% calculate mse for the above loss functions
mse_cauchy = 1/length(xData)*sum((yData - y_pred_cauchy).^2);
mse_fair = 1/length(xData)*sum((yData - y_pred_fair).^2);
mse_huber = 1/length(xData)*sum((yData - y_pred_huber).^2);
mse_talwar = 1/length(xData)*sum((yData - y_pred_talwar).^2);

% calculate mae for the above loss functions
mae_cauchy = 1/length(xData)*sum(abs(yData - y_pred_cauchy));
mae_fair = 1/length(xData)*sum(abs(yData - y_pred_fair));
mae_huber = 1/length(xData)*sum(abs(yData - y_pred_huber));
mae_talwar = 1/length(xData)*sum(abs(yData - y_pred_talwar));

figure(1)
scatter(xData,yData,'filled');  hold on 
plot(xData, y_pred_cauchy,'g') 
hold on;
plot(xData, y_pred_fair,'b') 
hold on;
plot(xData, y_pred_huber,'r')
hold on;
plot(xData, y_pred_talwar,'m')
hold on;
plot(xData, h_ols,'y');
lgd = legend('data','cauchy', 'fair', 'huber', 'talwar', 'ols');
lgd.Location = 'southeast';
ylabel('Predictions');
xlabel('xData');
title('OLS and Robust linear regression for all loss functions');