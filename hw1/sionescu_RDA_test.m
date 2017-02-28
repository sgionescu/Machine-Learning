function [Y_predict] = sionescu_RDA_test(X_test, RDAmodel, numofClass)
%
% Testing for RDA
%
% EC 503 Learning from Data
% Gaussian Discriminant Analysis
%
% Assuming D = dimension of data
% Inputs :
% X_test : test data matrix, each row is a test data point
% numofClass : number of classes 
% RDAmodel : the parameters of RDA classifier which has the following fields
% RDAmodel.Mu : numofClass * D matrix, i-th row = mean vector of class i
% RDAmodel.Sigmapooled : D * D  covariance matrix 
% RDAmodel.Pi : numofClass * 1 vector, Pi(i) = prior probability of class i
%
% Assuming that the classes are labeled  from 1 to numofClass
% Output :
% Y_predict predicted labels for all the testing data points in X_test



Sigma = RDAmodel.Sigmapooled;
Mu = RDAmodel.Mu;
Pi = RDAmodel.Pi;


for g = 1: size(Sigma, 3)
   for s = 1: numofClass   
        Mu1_rep = repmat(Mu(s,:),size(X_test,1),1);
        a = Mu1_rep/Sigma(:,:,g) * X_test' - (1/2)*Mu1_rep/Sigma(:,:,g) * Mu1_rep' + log(Pi(s));
        value_per_class(:,s) = diag(a);
   end
    

    max_val = max(value_per_class,[],2);

    %Y_predict = zeros(size(X_test,1),1);
    for z = 1:size(max_val,1)
       Y_predict(z,g) = find(value_per_class(z,:) == max_val(z,1))-1; 
    end

end

end
