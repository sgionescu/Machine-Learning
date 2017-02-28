
% Assignment3_1b.m
% Silvia Ionescu
% 10-2-2016

% Description:
% Data set data_iris.mat was used with X = 150 samples x 4 features
% and Y - classifications for this 150 samples.
% This set was divided into 100 training samples and 50 test samples. 
% The numbers were picked uniformlu at random. 
% For each split training, testing, and performance evaluation was 
% performed. 
 
clear; close all; clc;

load('data_iris.mat')

% calcualte the sample size and number of classes
samples = 1:size(X,1);
numofClass = Y(size(X,1),1);

% assign variable dimentions, to be used later in the code
split_mean = zeros(numofClass, size(X,2),10);

QDA_Mu_10avg = zeros(numofClass, size(X,2));
LDA_Mu_10avg = zeros(numofClass, size(X,2));

variance = zeros(numofClass, size(X,2),10);
QDA_var_avg = zeros(numofClass, size(X,2));
LDA_var_avg = zeros(1, size(X,2));

QDA_confusion = zeros(numofClass,numofClass,10);
LDA_confusion = zeros(numofClass,numofClass,10);

% generate 10 splits
for i=1:10
    % fixing the random seed
    s = RandStream('mt19937ar','Seed',i-1);
    
    % generate 100 random numbers between 1-150
    sample_train = randperm(s,size(X,1),100);
    
    % create the train set by using the 100 random numbers 
    X_train = X(sample_train,:);
    Y_train = Y(sample_train,:);

    % create the test set by taking the remaining samles 
    sample_test = setdiff(samples, sample_train);
    X_test = X(sample_test,:);
    Y_test = Y(sample_test,:);

    % QDA Model test and train
    QDAmodel = sionescu_QDA_train(X_train, Y_train, numofClass);
    QDA_Y_predict = sionescu_QDA_test(X_test, QDAmodel, numofClass);
    
    % LDA Model test and train
    LDAmodel = sionescu_LDA_train(X_train, Y_train, numofClass);
    LDA_Y_predict = sionescu_LDA_test(X_test, LDAmodel, numofClass);
    
    % add up the mean accross 10 splits for QDA and LDA
    QDA_Mu_10avg = QDA_Mu_10avg + QDAmodel.Mu;
    LDA_Mu_10avg = LDA_Mu_10avg + LDAmodel.Mu;
    
    % calcualte the QDA Sigma summed over 10 splits per class
    QDA_Sigma = QDAmodel.Sigma;
    for j = 1:numofClass
        variance(j,:,i) = diag(QDA_Sigma(:,:,j))';
    end
    QDA_var_avg = QDA_var_avg + variance(:,:,i);
    
    % calcualte the LDA Sigma summed over 10 splits 
    LDA_Sigma = diag(LDAmodel.Sigmapooled)';
    LDA_var_avg = LDA_var_avg + LDA_Sigma;
    
    % QDA confusion matrix
    QDA_confusion(:,:,i) = confusionmat(Y_test,QDA_Y_predict);
    QDA_CCR(1,i) = sum(diag(QDA_confusion(:,:,i)))/ sum(sum(QDA_confusion(:,:,i)));
    
    % LDA confusion matrix
    LDA_confusion(:,:,i) = confusionmat(Y_test,LDA_Y_predict);
    LDA_CCR(1,i) = sum(diag(LDA_confusion(:,:,i)))/ sum(sum(LDA_confusion(:,:,i)));
end

% 10 splits average mean per class for QDA and LDA
QDA_Mu_10avg = QDA_Mu_10avg/10;
LDA_Mu_10avg = LDA_Mu_10avg/10;

% 10 splits average variance per class for QDA(3x4), LDA(1x4)
QDA_var_avg = QDA_var_avg/10;
LDA_var_avg = LDA_var_avg/10;

% mean of all 10 test CCR's
QDA_CCR_mean = sum(QDA_CCR)/10;
LDA_CCR_mean = sum(LDA_CCR)/10;

% standard deviation accross all 10 CCR's
QDA_CCR_sd = std(QDA_CCR);
LDA_CCR_sd = std(LDA_CCR);




