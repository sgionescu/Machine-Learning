

% Problem 8_1c
clear; clc; close all;

input = load('BostonListing.mat');
latitude = input.latitude;
longitude = input.longitude;
neighbourhood = input.neighbourhood;

k = 3;
points_per_cluster = [500,500,500];
[D1, D1_label] = sample_circle(k, points_per_cluster );
[D2, D2_label] = sample_spiral(k, points_per_cluster );

[D3_theta D3_ro] = cart2pol(D1(:,1), D1(:,2));

% linear transform to [0,1]
D3_theta = mat2gray(D3_theta);
D3_ro = mat2gray(D3_ro);
D3 = [D3_theta, D3_ro];



rng(2);

% D3, k = 2
figure(1)
subplot(3,1,1)
[idx_k2, C_k2] = kmeans(D3, 2, 'Replicates', 20, 'Distance', 'cityblock');

plot(D3_theta(idx_k2==1),D3_ro(idx_k2==1),'r.','MarkerSize',10)
hold on
plot(D3_theta(idx_k2==2),D3_ro(idx_k2==2),'b.','MarkerSize',10)
plot(C_k2(:,1),C_k2(:,2),'kx',...
     'MarkerSize',12,'LineWidth',3)
title('D3, k = 2')
xlabel('Angle');
ylabel('Radius');

subplot(3,1,2)
[idx_k3, C_k3] = kmeans(D3, 3, 'Replicates', 20, 'Distance', 'cityblock');

plot(D3_theta(idx_k3==1),D3_ro(idx_k3==1),'r.','MarkerSize',10)
hold on
plot(D3_theta(idx_k3==2),D3_ro(idx_k3==2),'b.','MarkerSize',10)
hold on
plot(D3_theta(idx_k3==3),D3_ro(idx_k3==3),'g.','MarkerSize',10)
plot(C_k3(:,1),C_k3(:,2),'kx',...
     'MarkerSize',12,'LineWidth',3)
title('D3, k = 3')
xlabel('Angle');
ylabel('Radius');

subplot(3,1,3)
[idx_k4, C_k4] = kmeans(D3, 4, 'Replicates', 20, 'Distance', 'cityblock');
plot(D3_theta(idx_k4==1),D3_ro(idx_k4==1),'r.','MarkerSize',10)
hold on
plot(D3_theta(idx_k4==2),D3_ro(idx_k4==2),'b.','MarkerSize',10)
hold on
plot(D3_theta(idx_k4==3),D3_ro(idx_k4==3),'g.','MarkerSize',10)
hold on
plot(D3_theta(idx_k4==4),D3_ro(idx_k4==4),'k.','MarkerSize',10)
plot(C_k4(:,1),C_k4(:,2),'kx',...
     'MarkerSize',12,'LineWidth',3)
title('D3, k = 4')
xlabel('Angle');
ylabel('Radius');

% 81c_ii l2 distance

[~, ~, sumdk2] = kmeans(D3, 2, 'Replicates', 20, 'Distance', 'sqeuclidean');
[~, ~, sumdk3] = kmeans(D3, 3, 'Replicates', 20, 'Distance', 'sqeuclidean');
[~, ~, sumdk4] = kmeans(D3, 4, 'Replicates', 20, 'Distance', 'sqeuclidean');
