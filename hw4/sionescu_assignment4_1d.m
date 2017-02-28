% Assignment 4.1d
% 
% Description:
% Train a Naive Bayes classifier using MAP rule with (alpha -1) = 1/W
% Where W is the number of distict words in the vocabulary.

clear; close all; clc;

% train and test datasets
train_data = load('train.data');
train_label = load('train.label');


test_data = load('test.data');
test_label = load('test.label');

vocabulary = importdata('vocabulary.txt');
newsgrouplabels = textread('newsgrouplabels.txt', '%s');

% length of the vocabulary
W = length(vocabulary);

class_word_freq = zeros(W,20);

% calculate beta using the traing set
for t = 1:20 
    index = find(train_label == t);

    % calculate the # of documets per class
    docsfreq_per_class(t) = (index(size(index,1)) - index(1) + 1)/size(train_label,1);
    
     % pull out each document at a one at a time
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
    
    % calculate the beta paramenters using the Dirichlet prior
    beta(:,t) = (class_word_freq(:,t) + 1/W)/(words_per_class(t) + W*1/W);
end

beta = beta';
% calulate the total number of zero and nonzero beta parameters
beta_num = (find(beta == 0));


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
        label(l,1) = 1;
    else 
        label(l,1) = find(classification == max_prob);
    end

end

% calculate the confusion matrix, CCR, and error rate
confusion = confusionmat(test_label, label);
disp(confusion);
CCR = sum(diag(confusion))/ sum(sum(confusion));
error_rate = 1 - CCR
