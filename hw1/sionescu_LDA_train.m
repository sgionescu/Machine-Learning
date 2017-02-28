function [LDAmodel] = sionescu_LDA_train(X_train, Y_train, numofClass)
%
% Training LDA
%
% EC 503 Learning from Data
% Gaussian Discriminant Analysis
%
% Assuming D = dimension of data
% Inputs :
% X_train : training data matrix, each row is a training data point
% Y_train : training labels for rows of X_train
% numofClass : number of classes
%
% Assuming that the classes are labeled  from 1 to numofClass
% Output :
% LDAmodel : the parameters of LDA classifier which has the following fields
% LDAmodel.Mu : numofClass * D matrix, i-th row = mean vector of class i
% LDAmodel.Sigmapooled : D * D  covariance matrix
% LDAmodel.Pi : numofClass * 1 vector, Pi(i) = prior probability of class i
%

% Silvia Ionescu
% Date: 10-2-2016

% determine the length of the training set
X_train_size = size(X_train,1);

% pull out the features for class 1, 2, 3 and place it in a cell array
class1_X = {X_train(find(Y_train == 1),:),find(Y_train == 1)};
class2_X = {X_train(find(Y_train == 2),:),find(Y_train == 2)};
class3_X = {X_train(find(Y_train == 3),:),find(Y_train == 3)};
class = {class1_X, class2_X, class3_X};

% assign variable dimentions, to be used later in the code
Mu = zeros(numofClass, size(X_train,2));
%Sigma = zeros(size(X_train,2),size(X_train,2),numofClass);
Pi = zeros(numofClass,1);

% calculate mean vector, covariance matrix, and prior class probabiliy 
for c = 1: numofClass
    label = class{c};
    sublabel = label{1};
    sublabel_index = label{2};
    
    % iterate over features of the samples in a class
    for f = 1: size(sublabel, 2)        
        % calculate the mean
        Mu(c,f) = sum(sublabel(:,f))/size(sublabel,1);  
        
        % difference between the training samples and mean per class
        sublabel(:,f) = sublabel(:,f) - Mu(c,f);    
    end

    class_diff(:,c) = {sublabel, sublabel_index};
    
    % determine prior class probability
    Pi(c) = size(sublabel,1)/X_train_size;
end

% recombine classes into a sample set of the original training set size
sample_set_diff = [];
for i = 1:size(class_diff, 2)
    first = class_diff(1,i);
    second = class_diff(2,i);
    sample_cat = [first{1}, second{1}];
    sample_set_diff = [sample_set_diff;sample_cat];
end

% sort the set according to the original indices
sample_diff_sort = sortrows(sample_set_diff,size(sample_set_diff,2));

% calculate the covariance matrix
sample_diff_for_cov = sample_diff_sort(:,1:(end-1)); 
Sigmapooled = (sample_diff_for_cov'*sample_diff_for_cov)/(length(X_train)-numofClass); 

% build the LDAmodel struct
LDAmodel = struct('Mu', Mu, 'Sigmapooled', Sigmapooled, 'Pi', Pi);


end
