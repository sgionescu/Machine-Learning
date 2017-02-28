function [RDAmodel]= sionescu_RDA_train(X_train, Y_train,gamma, numofClass)
%
% Training RDA
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
% RDAmodel : the parameters of RDA classifier which has the following fields
% RDAmodel.Mu : numofClass * D matrix, i-th row = mean vector of class i
% RDAmodel.Sigmapooled : D * D  covariance matrix
% RDAmodel.Pi : numofClass * 1 vector, Pi(i) = prior probability of class i

% Silvia Ionescu
% 10-2-2016

% Finding Sigma LDA 
X_train_size = size(X_train,1);

class1_X = {X_train(find(Y_train == 0),:),find(Y_train == 0)};
class2_X = {X_train(find(Y_train == 1),:),find(Y_train == 1)};
class = {class1_X, class2_X};

% calculate mean
Mu = zeros(numofClass, size(X_train,2));
%C1 = zeros(size(class1_X,1), size(class1_X,2));
%Sigma = zeros(size(X_train,2),size(X_train,2),numofClass);
Pi = zeros(numofClass,1);

for c = 1: numofClass
    label = class{c};
    sublabel = label{1};
    sublabel_index = label{2};
    for f = 1: size(sublabel, 2)
        % 3x4 matrix - mean
        Mu(c,f) = sum(sublabel(:,f))/size(sublabel,1);  
        sublabel(:,f) = sublabel(:,f) - Mu(c,f);    
    end

    class_diff(:,c) = {sublabel, sublabel_index};
    Pi(c) = size(sublabel,1)/X_train_size;
end

sample_set_diff = [];
for i = 1:size(class_diff, 2)
    first = class_diff(1,i);
    second = class_diff(2,i);
    sample_cat = [first{1}, second{1}];
    sample_set_diff = [sample_set_diff;sample_cat];
end

sample_diff_sort = sortrows(sample_set_diff,size(sample_set_diff,2));
sample_diff_for_conv = sample_diff_sort(:,1:(end-1)); 
Sigma = (sample_diff_for_conv'*sample_diff_for_conv)/(length(X_train)-numofClass); 

% ------- Have Sigma LDA -----

% apply regularized discriminant analysis (RDA)
for g = 1: size(gamma,2)
    Sigmapooled(:,:,g) = gamma(g)*diag(diag(Sigma)) + (1-gamma(g))*Sigma;    
end
    
RDAmodel = struct('Mu', Mu, 'Sigmapooled', Sigmapooled, 'Pi', Pi);

end
