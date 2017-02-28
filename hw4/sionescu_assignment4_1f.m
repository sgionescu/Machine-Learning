% Assignment 4.1f
% 
% Description:
%
% Bayesian Naive Bayes with Dirichlet prior
% The most common words listed in the stopword.text file were removed 
% from the vocabulary, training set, and testing set. 
% The size of the new vocabulary was calculated, as well as the average 
% document length.
% Determined the error rate. 

clear; close all; clc;

% train and test datasets
train_data = load('train.data');
train_label = load('train.label');

test_data = load('test.data');
test_label = load('test.label');

vocabulary = textread('vocabulary.txt', '%s');
stoplist = textread('stoplist.txt', '%s');

% find the index for the stoplist words in vocabulary
stoplist = unique(stoplist);
p = 0;
for i =1: size(stoplist, 1);
    a = stoplist(i);
    word = a(1);
    index = strmatch(word, vocabulary, 'exact');
    if ~isempty(index)
        p = p + 1;
        remove_list(p,1) = index;
        
    end
end

% remove words from train set
train_index1 = [];
for i = 1:size(remove_list,1)
    train_index = find(train_data(:,2) == remove_list(i)); 
    train_index1 = [train_index1; train_index];
end
train_data_ind = 1:length(train_data);
train_index_diff = setdiff(train_data_ind, train_index1);
train_data = train_data(train_index_diff,:);

% average document length for training set
for i = 1:size(train_label, 1)
    train_doc_len = train_data(find(train_data(:,1) == i),:);
    train_doc_sum(i) = sum(train_doc_len(:,3));
    
end

average_train_len = sum(train_doc_sum)/length( train_doc_sum)

% remove words from test set
test_index1 = [];
for i = 1:size(remove_list,1)
    test_index = find(test_data(:,2) == remove_list(i)); 
    test_index1 = [test_index1; test_index];
end
test_data_ind = 1:length(test_data);
test_index_diff = setdiff(test_data_ind, test_index1);
test_data = test_data(test_index_diff,:);

% average document length for test set
for i = 1:size(test_label, 1)
    test_doc_len = test_data(find(test_data(:,1) == i),:);
    test_doc_sum(i) = sum(test_doc_len(:,3));
    
end

average_test_len = sum(test_doc_sum)/length(test_doc_sum)

% remove words from vocabulary
vocab_ind = 1:length(vocabulary);
vocab_diff = setdiff(vocab_ind, remove_list);
vocabulary_new = vocabulary(vocab_diff,:);


%%%%% From part 4_1d %%%%%%%%%%%%%%%
W = length(vocabulary);
W_new = length(vocabulary_new);

disp('New vocabulary length:');
disp(W_new);

class_word_freq = zeros(W,20);

% calculate beta using the new traing set
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
    beta(:,t) = (class_word_freq(:,t) + 1/W_new)/(words_per_class(t) + W_new*1/W_new);
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
    % if the probability is infinity for all beta's for the 20 classes
    % then make a random guess say 1, otherwise choose the class with
    % the max value
    max_prob = max(classification);
    if max_prob == -Inf
        label(l,1) = 1;
    else 
        label(l,1) = find(classification == max_prob);
    end

end

% calculate the confusion matrix, CCR, and error rate
confusion = confusionmat(test_label, label);
CCR = sum(diag(confusion))/ sum(sum(confusion));
error_rate = 1 - CCR



