
% Problem 7.1 Exploring Boston Housing Data with Regression Trees
% Silvia Ionecu
% 11-18-2016

clear; close all; clc;

% train and test datasets
housing_data = load('housing_data.mat');

train_data = housing_data.Xtrain;
train_label = housing_data.ytrain;

test_data = housing_data.Xtest;
test_label = housing_data.ytest;
feature_names = housing_data.feature_names;

t = classregtree(train_data,train_label, 'method', 'regression', 'minleaf', 20, 'names', feature_names);
view(t);

hw7_1b = [5 18 2.31 1 0.5440 2 64 3.7 1 300 15 390 10];

% Part b prediction
prediction_b = eval(t,hw7_1b);

% Part c
for i = 1:25
    t2 = classregtree(train_data,train_label, 'method', 'regression', 'minleaf', i, 'names', feature_names);
    predic_train_c = eval(t2,train_data);
    predic_test_c = eval(t2, test_data);
    err_train(i) = immse(predic_train_c,train_label);
    err_test(i) = immse(predic_test_c, test_label);
    
    mae_train(i) = 1/length(train_label)*sum(abs(train_label-predic_train_c));
    mae_test(i) = 1/length(test_label)*sum(abs(test_label-predic_test_c));
   
    
end

figure(2)
plot(1:25,  mae_train, 'g');
hold on;
plot(1:25, mae_test, 'r');
hold off;
legend('MAE train', 'MAE test');
xlabel('Observations per leaf');
ylabel('Training and testing MAE ');
title('Training and testing data MAE vs. observations per leaf');
