% Part 5.1c
% Predict the labels for the real test dataset
% After running part b a few times with lambda values between 
% 10^-5 and 10^5, I was able to norrow it down to a lambda value of ~ 1. 
% After this I took a range of values for lambda around 1: 
% 0.4, 0.8, 1, 4, 8
% Setteled on lambda = 1 with the best CCR and logloss. 
% Then used the training set to predict the labels for the test dataset

% Silvia Ionescu
% 10-25-2016
clear; close all; clc;

train = load('train_data.mat');
test = load('test_data.mat');

train_data = train.train_data;
train_label = train.label;
train_label = train_label';

test_data = test.test_data;

n = 10^-5;
lambda = 8;

% Initialize the parameters w to zero
w = zeros(39, size(train_data,2));

% loop over t = 1000 iterations
for i = 1:10
    disp(i);
    exponential = exp(w * train_data');
    exp_sum = sum(exponential,1); 
    
    penalty = zeros(39, size(train_data,1));
    
    % calculate NILL
    for k = 1:39
        
        penalty(k,(train_label == k)) = 1;  
        
        % for f(theta)
        partial = penalty(k,:)*train_data;
        NILL_b(k) = w(k,:)*partial';
         
        % euclidian distamce
        eclidian(k) = w(k,:)*w(k,:)';
        %eclidian(k) = norm(w(k,:));
    end
    
  exp_sum_rep = repmat(exp_sum,39,1);
    
  grad_a = (exponential./exp_sum_rep) - penalty;
  grad_NILL = grad_a * train_data;
    
  NILL_a = sum(log(exp_sum),2);  
    
    
  % calculate NILL
  NILL(i) = NILL_a - sum(NILL_b);
  f_theta(i) = NILL(i) + (lambda/2)* sum(eclidian);
  
  gradient_f_theta = grad_NILL + lambda*w;
  
  % update weights  
  w = w - n*gradient_f_theta;
    
  % predict the labels
  label_calc = w*test_data';
  [label_max, label_predicted] =  max(label_calc,[],1);

end

train_for_labels = load('data_SFcrime_train.mat');

train_category_crime = train_for_labels.Category;

first_colum = 1:size(label_predicted,2);
first_colum = first_colum';

% binary vectors of the labers
final_label = zeros(size(label_predicted,2),39);
for l = 1:size(label_predicted,2)
    disp(l);
    final_label(l,label_predicted(1,l)) = 1;
end


final = [first_colum, final_label];
csvwrite('test_label.csv',final);