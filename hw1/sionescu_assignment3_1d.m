
% Assignment3_1d.m
% Silvia Ionescu
% 10-2-2016

% Description:
% Data set data_cancer.mat was used with X = 216 samples x 4000 features
% and Y - classifications for this 216 samples.
% This set was divided into 150 training samples and 66 test samples. 
% The numbers were picked uniformlu at random. 
% Used Regularised Discriminant Analysis(RDA) because the covariance 
% matrix is singular. 
 

clear; close all; clc;

load('data_cancer.mat')

% calcualte the sample size and number of classes
samples = 1:size(X,1);

% fixing the random seed
s = RandStream('mt19937ar','Seed',5);

% generate 150 random numbers between 1-216
sample_train = randperm(s,size(X,1),150);

% create the train set by using the 150 random numbers 
X_train = X(sample_train,:);
Y_train = Y(sample_train,:);

% create the test set by taking the remaining samles
sample_test = setdiff(samples, sample_train);
X_test = X(sample_test,:);
Y_test = Y(sample_test,:);

% number of classes = 2
numofClass = 2;

% generate a gamma between 01. and 1
gamma = 0.1:0.05:1;

% RDA Model test and train
RDAmodel = sionescu_RDA_train(X_train, Y_train,gamma, numofClass);
RDA_Y_predict = sionescu_RDA_test(X_test, RDAmodel, numofClass);
RDA_Y_predict_train = sionescu_RDA_test(X_train, RDAmodel, numofClass);

for i = 1:size(RDA_Y_predict,2)
    % Confusion matrix for X_test
    confusion_mat_test(:,:,i) = confusionmat(RDA_Y_predict(:,i), Y_test);
    CCR_test(:,i) = sum(diag(confusion_mat_test(:,:,i)))/sum(sum(confusion_mat_test(:,:,i)));

    % Confusion matrix and CCR for X_train
    confusion_mat_train(:,:,i) = confusionmat(RDA_Y_predict_train(:,i), Y_train);
    CCR_train(:,i) = sum(diag(confusion_mat_train(:,:,i)))/sum(sum(confusion_mat_train(:,:,i)));
end

% plot CCRs for training and testing sets
figure(1);
plot(gamma, CCR_test);
title('Lambda vs. training and testing CCRs');
xlabel('Lambda');
ylabel('CCR');
axis([0.08 1.015 0 1.03]);
hold on;
plot(gamma, CCR_train, '--');
hold off;
legend ('CCR test set', 'CCR_training set','Location','southwest');