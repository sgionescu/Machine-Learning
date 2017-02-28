% Part 5.1b 
% Find the Objective function over 1000 iterations
% Find the CCR over 1000 iterations
% Find the Logloss over 1000 iterations
% The training set was split in two subsets:
%               - first 60% for training
%               - second 40% for 'testing'
% This code takes about ~ 30 min to run 

% Silvia Ionescu
% 10-25-2016


clear; close all; clc;

train = load('train_data.mat');

train_data = train.train_data;
train_label = train.label;
train_label = train_label';

% split the training dataset into 60% for training and 40% for 'testing' 
x = train_data(1:526830,:);
y = train_label(1:526830,:);

x_test = train_data(526831:end,:);
y_test = train_label(526831:end,:);

n = 10^-5;
lambda = 100;

% Initialize the parameters w to zero
w = zeros(39, size(train_data,2));

% loop over t = 1000 iterations
penalty = zeros(39, size(x,1));

for i = 1:1000
    disp(i);
    exponential = exp(w * x');
    exp_sum = sum(exponential,1); 
    
    penalty = zeros(39, size(x,1));
    for p = 1:39
        penalty(p,(y == p)) = 1;    
    end
    
    exp_sum_rep = repmat(exp_sum,39,1);
    
    grad_a = (exponential./exp_sum_rep) - penalty;
    grad_NILL = grad_a * x;
    
    NILL_a = sum(log(exp_sum),2);
    
    % calculate NILL
    for k = 1:39
        
        % for f(theta)
        test = penalty(k,:)*x;
        NILL_b(k) = w(k,:)*test';
         
        % euclidian distamce
        eclidian(k) = w(k,:)*w(k,:)';
        %eclidian(k) = norm(w(k,:));
    end
 
  % calculate NILL
  NILL(i) = NILL_a - sum(NILL_b);
  f_theta(i) = NILL(i) + (lambda/2)* sum(eclidian);
  
  gradient_f_theta = grad_NILL + lambda*w;
  
  % update weights using gradient descent  
  w = w - n*gradient_f_theta;
    
  % calculate CCR 
  label_calc = w*x_test';
  [label_max, label_predicted] =  max(label_calc,[],1);
    prob = 0;
    for c = 1:size(x_test,1)
        % for logloss calculation
        prob_prime = exp(w(y_test(c,1),:)* x_test(c,:)');
        
        if prob_prime < 10^-10
            prob_prime = 10^-10 ;
        end
        
        prob = prob + log(prob_prime); 
    end
    
    
    confusion_matrix = confusionmat(y_test, label_predicted);
    CCR(i) = sum(diag(confusion_matrix))/sum(sum(confusion_matrix));
    
    % logloss calculation 
    exponential_test = exp(w * x_test');
    exp_sum_test = sum(exponential_test,1); 
    exp_sum_test(find(exp_sum_test < 10^-10)) = 10^-10;
    
    exp_sum_test_log = sum(log(exp_sum_test));
    logloss(i) = -(1/size(x_test,1))*(prob - exp_sum_test_log);

end

figure(1);
plot(1:1000, f_theta);
title('Objective function vs. # of iterations');
xlabel('Iterations t');
ylabel('f_theta');

figure(2);
plot(1:1000, CCR);
title('CCRs vs. # of iterations');
xlabel('Iterations t');
ylabel('CCR');

figure(3);
plot(1:1000, logloss);
title('Test logloss vs. # of iterations');
xlabel('Iterations t');
ylabel('Logloss');

