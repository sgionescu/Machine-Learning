% Assignment 4.1e
% 
% Description:
% Bayesian Naive Bayes with Dirichlet prior.
% Choosing a range of (alpha -1) values. 

clear; close all; clc;

% train and test datasets
train_data = load('train.data');
train_label = load('train.label');


test_data = load('test.data');
test_label = load('test.label');

vocabulary = importdata('vocabulary.txt');
%vocabulary = textread('vocabulary.txt', '%d');

%newsgrouplabels = importdata('newsgrouplabels.txt');
newsgrouplabels = textread('newsgrouplabels.txt', '%s');

W = length(vocabulary);
% range of (alpha -1) values
alpha = [10^-5, 10^-4.5, 10^-4, 10^-3.5, 10^-3, 10^-2.5, 10^-2, 10^-1.5, 10^-1, 10^-0.5, 10^0, 10^.5, 10^1, 10^1.5];

% calculate the CCRs
for i = 1:length(alpha) 
    CCR(i) = naive_bayes(train_data, test_data, train_label, test_label, W , alpha(i));
end

% plot CCRs vs alpha-1
semilogx(alpha, CCR);
title ('Problem 4.1e - CCRs vs. alpha');
xlabel('(alpha -1)');
ylabel('CCRs');
