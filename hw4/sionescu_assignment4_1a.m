% Assignment 4.1a
% 
% Description:
% Determine the total number of unique words for the training set, 
% test set, and the entire set.
% Calculate the average document length for the training and testing sets.
% Total number of unique words that appear in the test set, but not
% in the training set.


clear; close all; clc;

% train and test datasets
train_data = load('train.data');
train_label = load('train.label');

test_data = load('test.data');
test_label = load('test.label');

% count total number of unique words in test, train and dataset
train_unique = unique(train_data(:,2));
train_unique_length = size(train_unique,1)

test_unique = unique(test_data(:,2));
test_unique_length = size(test_unique,1)

dataset  = [train_data; test_data];
dataset_unique = unique(dataset(:,2));
dataset_unique_length = size(dataset_unique,1)

% total number of unique words in test, but not train
test_no_train = setdiff(test_unique, train_unique);

test_no_tain_length = size(test_no_train,1)


%%%%%%%%% Average document length %%%%%%

for i = 1:size(train_label, 1)
    train_doc_len = train_data(find(train_data(:,1) == i),:);
    train_doc_sum(i) = sum(train_doc_len(:,3));
    
end

average_train_len = sum(train_doc_sum)/length( train_doc_sum)

for i = 1:size(test_label, 1)
    test_doc_len = test_data(find(test_data(:,1) == i),:);
    test_doc_sum(i) = sum(test_doc_len(:,3));
    
end

average_test_len = sum(test_doc_sum)/length(test_doc_sum)
