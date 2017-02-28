
% Assignemnt 6_1cd
% Train a binary SVM classifier using a linear kernel and C is scalar
% in order to mitigate an unbalenced dataset. 
% Calculate CV-CCR as in part a
% Calculate CV precision, recall, and F-score
% Report the best C values in terms of recall and F-score.
% Takes ~ 2hrs to run. 
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


% training - documets id that are part of class 17
training_class17 = find(train_label == 17);
training_classNot17 = setdiff(1:length(train_label), training_class17);

% labels for class 17
train_label17 = train_label(training_class17);
train_labelNot17 = train_label(training_classNot17);

% label the classes that are not 17 equal to class 1
train_labelNot17(:,1) = 1;

% calculate the word frequency for class 17
train_word_frequency17 = zeros(length(training_class17), length(vocabulary));
for i = 1:length(training_class17)
    b = find(train_data(:,1) == training_class17(i));
    samples = train_data(b,:);
   
    % total number of words in the document
    words_per_doc = sum(samples(:,3),1);
    for j = 1: size(samples,1)
        train_word_frequency17(i,samples(j,2)) = samples(j,3)/words_per_doc;
    end
end


% calculate the word frequency for class 1
train_word_frequencyNot17 = zeros(length(training_classNot17), length(vocabulary));
for i = 1:length(training_classNot17)
    d = find(train_data(:,1) == training_classNot17(i));
    samples_Not17 = train_data(d,:);
   
    % total number of words in the document
    words_per_doc = sum(samples_Not17(:,3),1);
    for j = 1: size(samples_Not17,1)
        train_word_frequencyNot17(i, samples_Not17(j,2)) =  samples_Not17(j,3)/words_per_doc;
    end
end

% conbine class 17 and not 17
train_word_frequency = [train_word_frequency17; train_word_frequencyNot17];
train_label_binary = [train_label17; train_labelNot17];


%%%%% Test %%%%

% test - documets id that are part of class 17
 test_class17 = find(test_label == 17);
 test_classNot17 = setdiff(1:length(test_label), test_class17);

% test - labels for class 17
test_label17 = test_label(test_class17);
test_labelNot17 = test_label(test_classNot17);

% label the classes that are not 17 equal to class 1
test_labelNot17(:,1) = 1;


% test - calculate the word frequency for class 17
test_word_frequency17 = zeros(length(test_class17), length(vocabulary));
for i = 1:length(test_class17)
    b = find(test_data(:,1) == test_class17(i));
    samples = test_data(b,:);
   
    % total number of words in the document
    words_per_doc = sum(samples(:,3),1);
    for j = 1: size(samples,1)
        test_word_frequency17(i,samples(j,2)) = samples(j,3)/words_per_doc;
    end
end


% test - calculate the word frequency for class 1
test_word_frequencyNot17 = zeros(length(test_classNot17), length(vocabulary));
for i = 1:length(test_classNot17)
    d = find(test_data(:,1) == test_classNot17(i));
    samples_Not17 = test_data(d,:);
   
    % total number of words in the document
    words_per_doc = sum(samples_Not17(:,3),1);
    for j = 1: size(samples_Not17,1)
        test_word_frequencyNot17(i, samples_Not17(j,2)) =  samples_Not17(j,3)/words_per_doc;
    end
end

% conbine class 17 and not 17
test_word_frequency = [test_word_frequency17; test_word_frequencyNot17];
test_label_binary = [test_label17; test_labelNot17];



vocab_index = 1:length(vocabulary);
train_index_diff = setdiff(vocab_index, remove_list);

% remove stoplist words from train_data
train_word_frequency = train_word_frequency(:,train_index_diff); 
clear train_word_frequency17 train_word_frequencyNot17 test_word_frequency17 test_word_frequencyNot17 train_data test_data train_label test_label

% remove stoplist words from test_data
test_word_frequency = test_word_frequency(:,train_index_diff);

clear remove_list samples samples_Not17 stoplist test_label17 test_labelNot17 train_index_diff train_label17 train_labelNot17
clear vocabulary vocab_index training_class17 training_classNot17 test_class17 test_classNot17

% split into 5 folds 
s = RandStream('mt19937ar','Seed',1);

random_train = randperm(s,size(train_word_frequency,1));

split1 = train_word_frequency(random_train(1:2253),:);
split2 = train_word_frequency(random_train(2254:4507),:);
split3 = train_word_frequency(random_train(4508:6761),:);
split4 = train_word_frequency(random_train(6762:9015),:);
split5 = train_word_frequency(random_train(9016:11269),:);


label_split1 = train_label_binary(random_train(1:2253),:);
label_split2 = train_label_binary(random_train(2254:4507),:);
label_split3 = train_label_binary(random_train(4508:6761),:);
label_split4 = train_label_binary(random_train(6762:9015),:);
label_split5 = train_label_binary(random_train(9016:11269),:);

C = [2^-5, 2^-4, 2^-3, 2^-2, 2^-1, 2^0, 2^1, 2^2, 2^3, 2^4, 2^5, 2^6, 2^7, 2^8, 2^9, 2^10, 2^11, 2^12, 2^13, 2^14, 2^15];

% calculate CCR over 5 folds for each C value
for i = 1:length(C) 
    a = tic
    disp(i);
    % test on split 5
    train1 = [split1; split2; split3; split4];
    label1 = [label_split1; label_split2; label_split3; label_split4];
    SVMStruct1 = svmtrain(train1, label1,'boxconstraint', C(i),'kernelcachelimit', 500000, 'autoscale', 'false');
    Group1 = svmclassify(SVMStruct1,split5);
    C1 = confusionmat(label_split5,Group1);
    CCR1 = sum(diag(C1))/ sum(sum(C1));
    
    C1 = C1';
    % calculate precision TP/np_hat
    P1 = C1(1,1)/ (C1(1,1) + C1(1,2));
    
    % calculate recal TP/np
    R1 = C1(1,1)/ (C1(1,1) + C1(2,1));
    % calculate F-score
    F1 = 2*(P1*R1)/(P1 + R1);
    
    clear train1 label1 SVMStruct1 Group1 C1
    
    % test on split 4
    train2 = [split1; split2; split3; split5];
    label2 = [label_split1;label_split2; label_split3; label_split5];
    SVMStruct2 = svmtrain(train2, label2,'boxconstraint',C(i), 'kernelcachelimit', 500000,'autoscale', 'false');
    Group2 = svmclassify(SVMStruct2, split4);
    C2 = confusionmat(label_split4,Group2);
    CCR2 = sum(diag(C2))/ sum(sum(C2));
    
    C2 = C2';
    % calculate precision TP/np_hat and recall
    P2 = C2(1,1)/ (C2(1,1) + C2(1,2));
    R2 = C2(1,1)/ (C2(1,1) + C2(2,1));
    F2 = 2*(P2*R2)/(P2 + R2);
    
    clear train2 label2 SVMStruct2 Group2 C2
    
    % test on split 3
    train3 = [split1; split2; split4; split5];
    label3 = [label_split1;label_split2; label_split4; label_split5];
    SVMStruct3 = svmtrain(train3, label3,'boxconstraint', C(i),'kernelcachelimit', 500000, 'autoscale', 'false');
    Group3 = svmclassify(SVMStruct3, split3);
    C3 = confusionmat(label_split3,Group3);
    CCR3 = sum(diag(C3))/ sum(sum(C3));

    C3 = C3';
     % calculate precision TP/np_hat
    P3 = C3(1,1)/ (C3(1,1) + C3(1,2));
    R3 = C3(1,1)/ (C3(1,1) + C3(2,1));
    F3 = 2*(P3*R3)/(P3 + R3);
    
    clear train3 label3 SVMStruct3 Group3 C3
    
    % test on split 2
    train4 = [split1; split3; split4; split5];
    label4 = [label_split1; label_split3; label_split4; label_split5];
    SVMStruct4 = svmtrain(train4, label4,'boxconstraint', C(i),'kernelcachelimit', 500000, 'autoscale', 'false');
    Group4 = svmclassify(SVMStruct4,split2);
    C4 = confusionmat(label_split2,Group4);
    CCR4 = sum(diag(C4))/ sum(sum(C4));

    C4 = C4';
    % calculate precision TP/np_hat
    P4 = C4(1,1)/ (C4(1,1) + C4(1,2));
    R4 = C4(1,1)/ (C4(1,1) + C4(2,1));
    F4 = 2*(P4*R4)/(P4 + R4);
    
    clear train4 label4 SVMStruct4 Group4 C4
    
    % test on split 1
    train5 = [split2; split3; split4; split5];
    label5 = [label_split2;label_split3; label_split4; label_split5];
    SVMStruct5 = svmtrain(train5, label5,'boxconstraint', C(i),'kernelcachelimit', 500000, 'autoscale', 'false');
    Group5 = svmclassify(SVMStruct5, split1);
    C5 = confusionmat(label_split1,Group5);
    CCR5 = sum(diag(C5))/ sum(sum(C5));
    
    C5 = C5';
     % calculate precision TP/np_hat
    P5 = C5(1,1)/ (C5(1,1) + C5(1,2));
    R5 = C5(1,1)/ (C5(1,1) + C5(2,1));
    F5 = 2*(P5*R5)/(P5 + R5);
    
    clear train5 label5  SVMStruct5 Group5 C5

    CCR(i) = (CCR1+CCR2+CCR3+CCR4+CCR5)/5  
    R(i) = (P1 + P2 + P3 + P4 + P5)/5
    P(i) = (R1 + R2 + R3 + R4 + R5)/5
    F(i) = (F1 + F2 + F3 + F4 + F5)/5
    toc(a)
end

figure(1)
plot(log2(C), CCR);
title('CV-CCRs as a function of C');
xlabel('log2 (C values)');
ylabel('CV-CCRs');

figure(2)
plot(log2(C), P,'r');
title('Precision, recall, and F-score as a function of C');
xlabel('log2 (C values)');
ylabel('Precision, recall, and F-score');
hold on;
plot(log2(C), R,'b');
hold on;
plot(log2(C), F, 'g');
legend('precision', 'recall', 'F-score');
hold off;


[CCR_max, CCR_max_index] = max(CCR);
C_max = C(CCR_max_index);

SVMStruct_all = svmtrain(train_word_frequency, train_label_binary,'boxconstraint', C_max,'kernelcachelimit', 500000, 'autoscale', 'false');
Group_all = svmclassify(SVMStruct_all, test_word_frequency);

confusion_matrix = confusionmat(test_label_binary,Group_all)
CCR_all = sum(diag(confusion_matrix))/ sum(sum(confusion_matrix));

[F_max, F_max_index] = max(F);
C_F_max = C(F_max_index);

% confusion matrix for F
SVMStruct_F = svmtrain(train_word_frequency, train_label_binary,'boxconstraint', C_F_max,'kernelcachelimit', 500000, 'autoscale', 'false');
Group_F = svmclassify(SVMStruct_F, test_word_frequency);

confusion_matrix_F = confusionmat(test_label_binary,Group_F)
CCR_F = sum(diag(confusion_matrix_F))/ sum(sum(confusion_matrix_F));

% confusion matrix for R
[R_max, R_max_index] = max(R);
C_R_max = C(R_max_index);

SVMStruct_R = svmtrain(train_word_frequency, train_label_binary,'boxconstraint', C_R_max,'kernelcachelimit', 500000, 'autoscale', 'false');
Group_R = svmclassify(SVMStruct_R, test_word_frequency);

confusion_matrix_R = confusionmat(test_label_binary,Group_R)
CCR_R = sum(diag(confusion_matrix_R))/ sum(sum(confusion_matrix_R));

save('hw6_1cd_results', 'P', 'R', 'F', 'CCR','C', 'confusion_matrix', 'CCR_all', 'confusion_matrix_F', 'confusion_matrix_R', 'C_R_max', 'C_F_max')
