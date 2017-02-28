function [QDAmodel]= sionescu_QDA_train(X_train, Y_train, numofClass)
%
% Training QDA
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
% QDAmodel : the parameters of QDA classifier which has the following fields
% QDAmodel.Mu : numofClass * D matrix, i-th row = mean vector of class i
% QDAmodel.Sigma : D * D * numofClass array, Sigma(:,:,i) = covariance matrix of class i
% QDAmodel.Pi : numofClass * 1 vector, Pi(i) = prior probability of class i

% Silvia Ionescu
% Date: 10-2-2016



% determine the length of the training set
X_train_size = size(X_train,1);

% pull out the features for class 1, 2, 3 and place it in a cell array
class1_X = X_train(find(Y_train == 1),:);
class2_X = X_train(find(Y_train == 2),:);
class3_X = X_train(find(Y_train == 3),:);
class = {class1_X, class2_X, class3_X};

% assign variable dimentions, to be used later in the code
Mu = zeros(numofClass, size(X_train,2));
Sigma = zeros(size(X_train,2),size(X_train,2),numofClass);
Pi = zeros(numofClass,1);

% calculate mean vector, covariance matrix, and prior class probabiliy 
for c = 1: numofClass
    label = class{c};
    
    % iterate over features of the samples in a class
    for f = 1: size(label,2)
        % calculate the mean
        Mu(c,f) = sum(label(:,f))/size(label,1);       
    end
    
    % calculate covariance per class
    Sigma(:,:,c) = cov(label); 
    
    % determine prior class probability
    Pi(c) = size(label,1)/X_train_size;
end

% build the QDAmodel struct
Pi = Pi';
QDAmodel = struct('Mu', Mu, 'Sigma', Sigma, 'Pi', Pi);

end
