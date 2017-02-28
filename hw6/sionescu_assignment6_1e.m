

% Assignemnt 6_1e
% One-vs-one (OVO) multiclass classification with linear kernel
% Train m(m-1)/2 binary SVM's for all class pairs.
% Calculate overall CCR 
% Calculate test and training time
% Silvia Ionescu
% 11-04-2016

clear; close all; clc;

% train and test datasets
train_data = load('train.data');
train_label = load('train.label');

vocabulary = textread('vocabulary.txt', '%s');
stoplist = textread('stoplist.txt', '%s');

test_data = load('test.data');
test_label = load('test.label');

% find the index for the stoplist words in vocabulary
stoplist = unique(stoplist);
p = 0;
for i =1: size(stoplist, 1);
    index = strmatch(stoplist(i), vocabulary, 'exact');
    if ~isempty(index)
        p = p + 1;
        remove_list(p,1) = index;       
    end
end

% set the words in stoplist to zero frequency
for i = 1: length(remove_list)
    train_loc = find(train_data(:,2) == remove_list(i));
    train_data(train_loc, 3) = 0;
    
    test_loc = find(test_data(:,2) == remove_list(i));
    test_data(test_loc, 3) = 0;
end

% find the word frequency of the training set
train_word_frequency = zeros(length(train_label), length(vocabulary));
for i = 1:length(train_label)
    disp(i)
    b = find(train_data(:,1) == i);
    samples = train_data(b,:);
   
    % total number of words in the document
    words_per_doc = sum(samples(:,3),1);
    for j = 1: size(samples,1)
        train_word_frequency(i,samples(j,2)) = samples(j,3)/words_per_doc;
    end
end
train_word_frequency = sparse(train_word_frequency);
clear train_data stoplist train_loc vocabulary



% find the word frequency ov the test set
test_word_frequency = zeros(length(test_label), 61188);
for i = 1:length(test_label)
    disp(i)
    b = find(test_data(:,1) == i);
    samples = test_data(b,:);
   
    % total number of words in the document
    words_per_doc = sum(samples(:,3),1);
    for j = 1: size(samples,1)
        test_word_frequency(i,samples(j,2)) = samples(j,3)/words_per_doc;
    end
end
test_word_frequency = sparse(test_word_frequency);
clear test_data b


vocab_index = 1:61188;
vocab_index_diff = setdiff(vocab_index, remove_list);

% remove stoplist words from train_data
train_word_frequency = train_word_frequency(:,vocab_index_diff);


% remove stoplist words from test_data
test_word_frequency = test_word_frequency(:,vocab_index_diff);

clear vocabulary stoplist

% determine label across 190 binary pairs
train_time = 0;
test_time = 0;
t = 0;
for i = 1:20
    
    class_a_index = find(train_label == i);
    class_a = train_word_frequency(class_a_index,:);
    class_label_a = train_label(class_a_index);
    
    for j = (i + 1):20
        t = t + 1;
        disp('t:')
        disp(t)
        % class j training data and labels
        class_b_index = find(train_label == j);
        class_b = train_word_frequency(class_b_index,:);
        class_label_b = train_label(class_b_index);
        
        train_data_binary = [class_a; class_b];
        train_label_binary = [class_label_a; class_label_b];
        
        tic
        SVMStruct = svmtrain(train_data_binary, train_label_binary,'kernelcachelimit', 1000000, 'autoscale', 'false');
        train_time = train_time + toc;
        
        tic
        Group(:,t) = svmclassify(SVMStruct, test_word_frequency);
        test_time = test_time + toc;
        
    end

end



label = mode(Group,2);
confusion_matrix = confusionmat(label,test_label)
CCR = sum(diag(confusion_matrix))/ sum(sum(confusion_matrix))

save('hw6_1e_results.mat', 'confusion_matrix', 'CCR')