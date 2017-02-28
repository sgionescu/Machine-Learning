function [Y_predict] = sionescu_LDA_test(X_test, LDAmodel, numofClass)
%
% Testing for LDA
%
% EC 503 Learning from Data
% Gaussian Discriminant Analysis
%
% Assuming D = dimension of data
% Inputs :
% X_test : test data matrix, each row is a test data point
% numofClass : number of classes 
% LDAmodel : the parameters of LDA classifier which has the follwoing fields
% LDAmodel.Mu : numofClass * D matrix, i-th row = mean vector of class i
% LDAmodel.Sigmapooled : D * D  covariance matrix
% LDAmodel.Pi : numofClass * 1 vector, Pi(i) = prior probability of class i
%
% Assuming that the classes are labeled  from 1 to numofClass
% Output :
% Y_predict predicted labels for all the testing data points in X_test

% Silvia Ionescu
% Date: 10-2-2016

Sigma = LDAmodel.Sigmapooled;
Mu = LDAmodel.Mu;
Pi = LDAmodel.Pi;

% inverse of the covariance matrix
Sigma_inv = inv(Sigma);

% calculating the LDADA model
for j = 1:size(X_test,1)
    for s = 1: numofClass
        a(j,s,:) = Mu(s,:) * Sigma_inv * X_test(j,:)' - (1/2)*Mu(s,:) * Sigma_inv * Mu(s,:)' + log(Pi(s));
    end
end

% finding the min value of a
max_val = max(a,[],2);

% determine test sample classification  
Y_predict = zeros(size(X_test,1),1);
for z = 1:size(max_val,1)
   Y_predict(z,:) = find(a(z,:) == max_val(z,1)); 
end

end
