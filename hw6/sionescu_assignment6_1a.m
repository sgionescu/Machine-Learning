
% Assignemnt 6_1a
% Train a binary SVM classifier using a linear kernel
% Choose C values between 2^-5 and 2^15
% Calculate CV-CCR as a function of C values
% Choose the best C value for the best CCR
% 
% Silvia Ionescu
% 11-04-2016

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

% documets id that are part of class1 and 20
training_class1 = find(train_label == 1);
training_class20 = find(train_label == 20);
index_class1_20 = [training_class1; training_class20];

% labels for class1 and class20
train_label1_20 = train_label(index_class1_20);

% calculate train dataset word frequency 
train_word_frequency = zeros(length(index_class1_20), length(vocabulary));
for i = 1:length(index_class1_20)
    b = find(train_data(:,1) == index_class1_20(i));
    samples = train_data(b,:);
   
    % total number of words in the document
    words_per_doc = sum(samples(:,3),1);
    for j = 1: size(samples,1)
        train_word_frequency(i,samples(j,2)) = samples(j,3)/words_per_doc;
    end
end

% test for class1 and class20
test_class1 = find(test_label == 1);
test_class20 = find(test_label == 20);
index_test_class1_20 = [test_class1; test_class20] ;

% labels for class1 and class20
test_label1_20 = test_label(index_test_class1_20);

test_word_frequency = zeros(length(index_test_class1_20), length(vocabulary));
for k = 1:length(index_test_class1_20)
    test_data_index = find(test_data(:,1)== index_test_class1_20(k));
    test_samples = train_data(test_data_index,:);
    
    % total number of words in the document
    test_words_per_doc = sum(test_samples(:,3),1);
    for l = 1:size(test_samples)
        test_word_frequency(k,test_samples(l,2)) = test_samples(l,3)/test_words_per_doc;
    end
end


vocab_index = 1:length(vocabulary);
train_index_diff = setdiff(vocab_index, remove_list);

% remove stoplist words from train_data
train_word_frequency_new = train_word_frequency(:,train_index_diff); 

% remove stoplist words from test_data
test_word_frequency_new = test_word_frequency(:,train_index_diff);

% split into 5 folds 
s = RandStream('mt19937ar','Seed',1);

random_train = randperm(s,size(train_word_frequency_new,1));
split1 = train_word_frequency_new(random_train(1:171),:);
split2 = train_word_frequency_new(random_train(172:342),:);
split3 = train_word_frequency_new(random_train(343:513),:);
split4 = train_word_frequency_new(random_train(514:684),:);
split5 = train_word_frequency_new(random_train(685:856),:);

label_split1 = train_label1_20(random_train(1:171),:);
label_split2 = train_label1_20(random_train(172:342),:);
label_split3 = train_label1_20(random_train(343:513),:);
label_split4 = train_label1_20(random_train(514:684),:);
label_split5 = train_label1_20(random_train(685:856),:);

C = [2^-5, 2^-4, 2^-3, 2^-2, 2^-1, 1, 2, 2^2, 2^3, 2^4, 2^5, 2^6, 2^7, 2^8, 2^9, 2^10, 2^11, 2^12, 2^13, 2^14, 2^15];
C = C';
% calculate CCR over 5 folds for each C valiue
for i = 1:length(C) 
    disp(i);
    % test on split 5
    train1 = [split1; split2; split3; split4];
    label1 = [label_split1; label_split2; label_split3; label_split4];
    SVMStruct1 = svmtrain(train1, label1,'boxconstraint', C(i)*ones(size(train1,1),1), 'autoscale', 'false');
    Group1 = svmclassify(SVMStruct1,split5);
    C1 = confusionmat(label_split5,Group1);
    CCR1 = sum(diag(C1))/ sum(sum(C1));

    % test on split 4
    train2 = [split1; split2; split3; split5];
    label2 = [label_split1;label_split2; label_split3; label_split5];
    SVMStruct2 = svmtrain(train2, label2,'boxconstraint',C(i)*ones(size(train2,1),1), 'autoscale', 'false');
    Group2 = svmclassify(SVMStruct2,split4);
    C2 = confusionmat(label_split4,Group2);
    CCR2 = sum(diag(C2))/ sum(sum(C2));

    % test on split 3
    train3 = [split1; split2; split4; split5];
    label3 = [label_split1;label_split2; label_split4; label_split5];
    SVMStruct3 = svmtrain(train3, label3,'boxconstraint', C(i)*ones(size(train3,1),1), 'autoscale', 'false');
    Group3 = svmclassify(SVMStruct3,split3);
    C3 = confusionmat(label_split3,Group3);
    CCR3 = sum(diag(C3))/ sum(sum(C3));

    % test on split 2
    train4 = [split1; split3; split4; split5];
    label4 = [label_split1; label_split3; label_split4; label_split5];
    SVMStruct4 = svmtrain(train4, label4,'boxconstraint', C(i)*ones(size(train4,1),1), 'autoscale', 'false');
    Group4 = svmclassify(SVMStruct4,split2);
    C4 = confusionmat(label_split2,Group4);
    CCR4 = sum(diag(C4))/ sum(sum(C4));

    % test on split 1
    train5 = [split2; split3; split4; split5];
    label5 = [label_split2;label_split3; label_split4; label_split5];
    SVMStruct5 = svmtrain(train5, label5,'boxconstraint', C(i)*ones(size(train5,1),1), 'autoscale', 'false');
    Group5 = svmclassify(SVMStruct5,split1);
    C5 = confusionmat(label_split1,Group5);
    CCR5 = sum(diag(C5))/ sum(sum(C5));

    CCR(i) = (CCR1+CCR2+CCR3+CCR4+CCR5)/5;    
end

figure(1)
plot(log2(C), CCR);
title('CV-CCRs as a function of C');
xlabel('log2 (C values)');
ylabel('CV-CCRs');

% pick the best C value 
[CCR_max, CCR_max_index] = max(CCR);
C_max = C(CCR_max_index);

% train SVM on all traing data 
SVMStruct_all = svmtrain(train_word_frequency_new, train_label1_20,'boxconstraint', C_max*ones(size(train_word_frequency_new,1),1), 'autoscale', 'false');
Group_all = svmclassify(SVMStruct_all,test_word_frequency_new);

confusion_matrix = confusionmat(test_label1_20,Group_all);
CCR_all = sum(diag(confusion_matrix))/ sum(sum(confusion_matrix));
