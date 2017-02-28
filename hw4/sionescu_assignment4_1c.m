% Assignment 4.1c
% 
% Description:
% Remove the words that only appear in the test documents, but not in the 
% training documents. 
% Repeat the testing and traing using the MLE rule like in Part 4_1b


clear; close all; clc;

% train and test datasets
train_data = load('train.data');
train_label = load('train.label');

test_data = load('test.data');
test_label = load('test.label');

vocabulary = textread('vocabulary.txt', '%s');

% find unique words 
train_unique = unique(train_data(:,2));
test_unique = unique(test_data(:,2));
dataset  = [train_data; test_data];

dataset_unique = unique(dataset(:,2));

test_no_train = setdiff(test_unique, train_unique);

index1 = [];
% remove words that apear only in test
for i = 1:size(test_no_train,1)
    index = find(test_data(:,2) == test_no_train(i)); 
    index1 = [index1; index];
end
test_data_ind = 1:length(test_data);
index_diff = setdiff(test_data_ind, index1);
test_data = test_data(index_diff,:);

% vocab_length = 53975
vocab_length = length(vocabulary) - length(test_no_train);

%%%%%%%%%%%%%%%%%  Apply MLE rule similar to Part 4.1b %%%%%%%%%%

class_word_freq = zeros(vocab_length,20);

% calculate beta using the traing set
for t = 1:20 
    index = find(train_label == t);

    % calculate the # of documets per class
    docsfreq_per_class(t) = (index(size(index,1)) - index(1) + 1)/size(train_label,1);
    
    % pull out each document at a time
    data_start = find(train_data(:,1) == index(1)); 
    data_stop = find(train_data(:,1) == index(size(index,1))); 
    start = data_start(1);
    stop = data_stop(size(data_stop,1));
    
    class = train_data(start:stop,:); 
    
    % calcualte the number of words per class 
    words_per_class(t) = sum(class(:,3));
    for s = 1: size(class, 1)
        class_word_freq(class(s,2),t) = class_word_freq(class(s,2),t) + class(s,3);
    end
    % calculate the beta paramenters
    beta(:,t) = class_word_freq(:,t)/words_per_class(t);
end

beta = beta';

% calulate the total number of zero and nonzero beta parameters
beta_zero = (find(beta == 0));
non_zero_beta = size(beta,1)*size(beta,2)- size(beta_zero);

disp('Non zero beta number:')
disp(non_zero_beta(1,1));

num_prob = 0;
%  from 1 to 7505 number of docs in test data
beta_sq_prod = ones(20,1);
for l = 1:test_data(length(test_data),1)
    % pull out each document of the test set
    test_samples = test_data((test_data(:,1)== l),:);
    
    % loop over the 20 beta paramenters 
    for n = 1:20 
    first = log(docsfreq_per_class(n));
    first2 = docsfreq_per_class(n);
    second = 0;
    second2 = 1;
        % go through the samples of each document
        for m = 1:size(test_samples,1)
            second  = second + test_samples(m,3)* log(beta(n,test_samples(m,2))) ;
            second2 = second2 * beta(n,test_samples(m,2))^test_samples(m,3);
        end
    % calculate the probability for each class    
    classification(n) = first + second;
    probability(n) = first2 * second2;
    end
    
     % find the number when max for all classes has P(Y=c|x) = 0
    if max(probability) == 0
        num_prob = num_prob + 1;
    end
    
    % determine the classification for the document
    % if the probability is infinity for all beta's for the 20 class
    % then make a random guess, otherwise choose the class with the max
    % value
    max_prob = max(classification);
    if max_prob == -Inf
        s = RandStream('mt19937ar','Seed',1);
        label(l,1) = randperm(s, 20, 1);
    else 
        label(l,1) = find(classification == max_prob);
    end

end

% calculate the confusion matrix, CCR, and error rate
confusion = confusionmat(test_label, label);
CCR = sum(diag(confusion))/ sum(sum(confusion));
error_rate = 1 - CCR

