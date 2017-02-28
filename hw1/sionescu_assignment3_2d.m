
% Assignment3_2d.m
% Silvia Ionescu
% 10-2-2016

% Description: Nearest Neighbor Classifier
% Problem 3.2d
% Apply 1-NNC to the two loaded datasets and display the CCRs 


clear; close all; clc;

% train and test datasets
load('data_mnist_train.mat');
load('data_mnist_test.mat');

% calculating <x,x> for the traing set
X_train_norm = sum(X_train.^2,2);
X_train_norm1 = X_train_norm*ones(1,1000);
X_train_norm1 = X_train_norm1';


c = 0;
for i = 1:1000:size(X_test,1)
    c = c + 1;
    X_test_part = X_test(i:(i+999),:); 
    
    % calculating <x',x'> for the test set
    X_test_norm = sum(X_test_part.^2,2);
    X_test_norm1 = X_test_norm * ones(1,60000);
    
    % calculating distance
    distance =  X_test_norm1 -2*(X_test_part*X_train') + X_train_norm1;
    [M,I(:,c)] = min(distance,[],2);
    class(:,c) = Y_train(I(:,c),1);
end

% determine confusion matrix and CCR
sample_class = class(:); 
confusion_matrix = confusionmat(sample_class, Y_test);
CCR = sum(diag(confusion_matrix))/sum(sum(confusion_matrix));
disp(['Problem3.2d CCR: ', num2str(CCR)])