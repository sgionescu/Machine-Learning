function [Y_predict] = sionescu_QDA_test(X_test, QDAmodel, numofClass)
%
% Testing for QDA
%
% EC 503 Learning from Data
% Gaussian Discriminant Analysis
%
% Assuming D = dimension of data
% Inputs :
% X_test : test data matrix, each row is a test data point
% numofClass : number of classes 
% QDAmodel: the parameters of QDA classifier which has the following fields
% QDAmodel.Mu : numofClass * D matrix, i-th row = mean vector of class i
% QDAmodel.Sigma : D * D * numofClass array, Sigma(:,:,i) = covariance
% matrix of class i
% QDAmodel.Pi : numofClass * 1 vector, Pi(i) = prior probability of class i
% 
% Assuming that the classes are labeled  from 1 to numofClass
% Output :
% Y_predict predicted labels for all the testing data points in X_test

% Silvia Ionescu
% Date: 10-2-2016

% Input data mean, covariance matrix, and prior probability
Sigma = QDAmodel.Sigma;
Mu = QDAmodel.Mu;
Pi = QDAmodel.Pi;

% inverse of the covariance matrix
for i = 1: numofClass
    Sigma_inv(:,:,i) = inv(Sigma(:,:,i));
end 

% assign variable dimentions, to be used later in the code
a = zeros(size(X_test,1), numofClass, 1);

% applying the QDA model
for j = 1:size(X_test,1)
    for s = 1: numofClass
        a(j,s,:) = (1/2)*(X_test(j,:) - Mu(s,:)) * Sigma_inv(:,:,s) * (X_test(j,:) - Mu(s,:))' + (1/2)*log(det(Sigma(:,:,s))) - log(Pi(s));
    end
end

% finding the min value 
min_val = min(a,[],2);
Y_predict = zeros(size(X_test,1),1);

% determine test sample classification  
for z = 1:size(min_val,1)
   Y_predict(z,:) = find(a(z,:) == min_val(z,1)); 
end

end
