
% Assignment3_2abc.m
% Silvia Ionescu
% 10-2-2016

% Description: Nearest Neighbor Classifier
% Problem 3.2a
% Fig 1: Scatter plot of all the training data in data_knnSimulation.mat
%
% Problem 3.2b
% Fig.2 shows the probabilities of being class 2 using k = 10
% Fig.3 shows the probabilities of being class 3 using k = 10
%
% Problem 3.2c
% Fig. 4 prediction of class using k = 1 NN
% Fig. 5 prediction of class using k = 5 NN

clear; close all; clc;

load('data_knnSimulation.mat')

% plot of all the training data in data_knnSimulation.mat
figure(1);
gscatter(Xtrain(:,1), Xtrain(:,2), ytrain,[1 0 0; 0 1 0; 0 0 1]);
legend('Feature 1', 'Feature 2', 'Clasification','Location','southeast');
title('Problem 3.2a - trainig data plot');
xlabel('Feature1');
ylabel('Feature2');

% create a 2D 96x96 matrix of test points 
x = -3.5:.1:6;
y = -3:.1:6.5;
[A B] = meshgrid(x,y);


class1 = 0;
class2 = 0;
class3 = 0;
c1 = 1;
c2 = 2;
c3 = 3;
for i = 1:length(A);
    for j = 1:length(B);
        for k = 1:size(Xtrain,1)
            % calculate Euclidian distance
            d = sqrt((A(i,j)-Xtrain(k,1))^2 + (B(i,j)-Xtrain(k,2))^2);
            class = ytrain(k,1);
            % contains distance and the according class of the neighbor
            dis_class(k,:) = [d class];      
        end
        
        % sort the array for each point in the 96x96 grid
        euclid_dis = sortrows(dis_class,1);
        % keep only the shorthest 10 distances and their class
        euclid_dis = euclid_dis(1:10,:);
        
        % calculate the class for k = 10 NNC
        class1 = 0;
        class2 = 0;
        class3 = 0;
        for p = 1:10
            test = euclid_dis(p,2);
            if (euclid_dis(p,2) == c1)
                class1 = class1 + 1;
            elseif (euclid_dis(p,2) == c2)
                class2 = class2 + 1;
            else
                
                class3 = class3 + 1;
            end
                
        end
        
        % probabilities of the nearest neighbors in terms of classes
        prob_10nn = [class1 c1; class2 c2; class3 c3];
        class_sort = sortrows(prob_10nn,2);
        prob_class2(i,j) = class_sort(2,1)/10;
        prob_class3(i,j) = class_sort(3,1)/10;
        
        % determine k = 1 NN classifier
        classification_k1(i,j) = euclid_dis(1,2);
        
        % calculate the k = 5 majority of neighbors
        freq1_5nn = 0;
        freq2_5nn = 0;
        freq3_5nn = 0;
        
        for t = 1:5
            if (euclid_dis(t,2) == c1)
                freq1_5nn = freq1_5nn + 1;
            elseif (euclid_dis(t,2) == c2)
                freq2_5nn = freq2_5nn + 1;
            else  
                freq3_5nn = freq3_5nn + 1;
            end
        end
        
        freq_5nn = [freq1_5nn c1; freq2_5nn c2; freq3_5nn c3];
        freq_5nn_sort = sortrows(freq_5nn, 1);
        classification_k5(i,j) = freq_5nn_sort(3,2);
       
    end
end

% plot the probabilities of being class 2 using k = 10
figure(2);
imagesc(prob_class2);
colormap jet;
axis('xy');
title('Problem 3.2b: p(y=2|data,k=10)');
xlabel('Feature1');
ylabel('Feature2');
colorbar;

% plot the probabilities of being class 3 using k = 10
figure(3);
imagesc(prob_class3);
colormap jet;
axis('xy');
title('Problem 3.2b: p(y=3|data,k=10)');
xlabel('Feature1');
ylabel('Feature2');
colorbar;

% plot the prediction of class using k = 1 NN
figure(4);
contourf(classification_k1);
map = [1 0 0; 0 1 0; 0 0 1];
colormap(map);
title('Problem 3.2c: k=1 Nearest Neighbor');
xlabel('Feature1');
ylabel('Feature2');

% plot the prediction of class using k = 5 NN
figure(5);
contourf(classification_k5);
map = [1 0 0; 0 1 0; 0 0 1];
colormap(map);
title('Problem 3.2c: k=5 Nearest Neighbor');
xlabel('Feature1');
ylabel('Feature2');




