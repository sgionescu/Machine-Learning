
% Problem 8_1a

clear; clc; close all;

input = load('BostonListing.mat');
latitude = input.latitude;
longitude = input.longitude;
neighbourhood = input.neighbourhood;

k = 3;
points_per_cluster = [500,500,500];
[D1, D1_label] = sample_circle(k, points_per_cluster );
[D2, D2_label] = sample_spiral(k, points_per_cluster );

% D = pdist2(D1(1,:),D2(1,:), 'euclidean');

rng(2);
% D1, k = 2
figure(1)
subplot(2,3,1)
[idx, C, sumd_D1_k2] = kmeans(D1, 2, 'Replicates', 20, 'Distance', 'sqeuclidean');
% gscatter(D1(:,1),D1(:,2), idx, 'br','xo')

plot(D1(idx==1,1),D1(idx==1,2),'r.','MarkerSize',10)
hold on
plot(D1(idx==2,1),D1(idx==2,2),'b.','MarkerSize',10)
plot(C(:,1),C(:,2),'kx',...
     'MarkerSize',12,'LineWidth',3)
title('D1, k = 2')
 
subplot(2,3,2)
[idx_D1_k3, C_D1_k3, sumd_D1_k3] = kmeans(D1, 3, 'Replicates', 20, 'Distance', 'sqeuclidean');
% gscatter(D1(:,1),D1(:,2), idx, 'brg','xo+')
plot(D1(idx_D1_k3==1,1),D1(idx_D1_k3==1,2),'r.','MarkerSize',10)
hold on
plot(D1(idx_D1_k3==2,1),D1(idx_D1_k3==2,2),'b.','MarkerSize',10)
hold on
plot(D1(idx_D1_k3==3,1),D1(idx_D1_k3==3,2),'g.','MarkerSize',10)
plot(C_D1_k3(:,1),C_D1_k3(:,2),'kx',...
     'MarkerSize',12,'LineWidth',3)
title('D1, k = 3')

subplot(2,3,3)
[idx_D1_k4, C_D1_k4, sumd_D1_k4] = kmeans(D1, 4, 'Replicates', 20, 'Distance', 'sqeuclidean');
% gscatter(D1(:,1),D1(:,2), idx, 'brgk','xo+*')
plot(D1(idx_D1_k4 == 1,1), D1(idx_D1_k4 == 1,2),'r.','MarkerSize',10)
hold on
plot(D1(idx_D1_k4 == 2,1), D1(idx_D1_k4 == 2,2),'b.','MarkerSize',10)
hold on
plot(D1(idx_D1_k4 == 3,1),D1(idx_D1_k4 == 3,2),'g.','MarkerSize',10)
hold on
plot(D1(idx_D1_k4 == 4,1),D1(idx_D1_k4 == 4,2),'k.','MarkerSize',10)
plot(C_D1_k4(:,1),C_D1_k4(:,2),'mx',...
     'MarkerSize',12,'LineWidth',3)
title('D1, k = 4')


subplot(2,3,4)
[idx_D2_k2, C_D2_k2, sumd_D2_k2] = kmeans(D2, 2, 'Replicates', 20, 'Distance', 'sqeuclidean');
% gscatter(D1(:,1),D1(:,2), idx, 'br','xo')
plot(D2(idx_D2_k2==1,1),D2(idx_D2_k2==1,2),'r.','MarkerSize',10)
hold on
plot(D2(idx_D2_k2==2,1),D2(idx_D2_k2==2,2),'b.','MarkerSize',10)
plot(C_D2_k2(:,1),C_D2_k2(:,2),'kx',...
     'MarkerSize',12,'LineWidth',3)
title('D2, k = 2')

subplot(2,3,5)
[idx_D2_k3, C_D2_k3, sumd_D2_k3] = kmeans(D2, 3, 'Replicates', 20, 'Distance', 'sqeuclidean');
% gscatter(D1(:,1),D1(:,2), idx, 'brg','xo+')
plot(D2(idx_D2_k3==1,1),D2(idx_D2_k3==1,2),'r.','MarkerSize',10)
hold on
plot(D2(idx_D2_k3==2,1),D2(idx_D2_k3==2,2),'b.','MarkerSize',10)
hold on
plot(D2(idx_D2_k3==3,1),D2(idx_D2_k3==3,2),'g.','MarkerSize',10)
plot(C_D2_k3(:,1),C_D2_k3(:,2),'kx',...
     'MarkerSize',12,'LineWidth',3)
title('D2, k = 3')

subplot(2,3,6)
[idx_D2_k4, C_D2_k4, sumd_D2_k4] = kmeans(D2, 4, 'Replicates', 20, 'Distance', 'sqeuclidean');
% gscatter(D1(:,1),D1(:,2), idx, 'brgk','xo+*')

plot(D2(idx_D2_k4==1,1),D2(idx_D2_k4==1,2),'r.','MarkerSize',10)
hold on
plot(D2(idx_D2_k4==2,1),D2(idx_D2_k4==2,2),'b.','MarkerSize',10)
hold on
plot(D2(idx_D2_k4==3,1),D2(idx_D2_k4==3,2),'g.','MarkerSize',10)
hold on
plot(D2(idx_D2_k4==4,1),D2(idx_D2_k4==4,2),'k.','MarkerSize',10)
plot(C_D2_k4(:,1),C_D2_k4(:,2),'mx',...
     'MarkerSize',12,'LineWidth',3)
title('D2, k = 4')


% 
% % part 8.1a part ii
% 
% test = D1(idx==1,:);
% C_test = C(1, :);
% distance = pdist2(test,C_test, 'euclidean');

% part 8.1b 
sigma = 0.2;
D1_distance = pdist2(D1, D1, 'euclidean');
D1_similarity_matrix = exp(-((D1_distance.^2)/(2*sigma^2))); 
D1_W = D1_similarity_matrix;
D1_degree_matrix = sum(D1_W, 2);
D1_degree_matrix = diag(D1_degree_matrix);

% D1 -calculate Laplacians
D1_L_unnorm = D1_degree_matrix - D1_W;
D1_L_rw = (D1_degree_matrix^-1) * D1_L_unnorm;
D1_L_sym = (D1_degree_matrix^-(1/2)) * D1_L_unnorm * (D1_degree_matrix^-(1/2));

% D1 - calculate eigenvalues
 
D1_L_unnorm_eigval = sort(eig(D1_L_unnorm), 'ascend');
D1_L_rw_eigval = sort(eig(D1_L_rw), 'ascend');
D1_L_sym_eigval = sort(eig(D1_L_sym), 'ascend');

% D2
D2_distance = pdist2(D2, D2, 'euclidean');
D2_similarity_matrix = exp(-((D2_distance.^2)/(2*sigma^2))); 
D2_W = D2_similarity_matrix;
D2_degree_matrix = sum(D2_W, 2);
D2_degree_matrix = diag(D2_degree_matrix);

% D1 -calculate Laplacians
D2_L_unnorm = D2_degree_matrix - D2_W;
D2_L_rw = (D2_degree_matrix^-1) * D2_L_unnorm;
D2_L_sym = (D2_degree_matrix^-(1/2)) * D2_L_unnorm * (D2_degree_matrix^-(1/2));

% D1 - calculate eigenvalues
 
D2_L_unnorm_eigval = sort(eig(D2_L_unnorm), 'ascend');
D2_L_rw_eigval = sort(eig(D2_L_rw), 'ascend');
D2_L_sym_eigval = sort(eig(D2_L_sym), 'ascend');


% 8.1bi plot
figure(2)
subplot(3,3,1);
plot(D1_L_unnorm_eigval);
title('D1 Lunnorm eigenvalues');
subplot(3,3,2);
plot(D1_L_rw_eigval);
title('D1 Lrw eigenvalues');
subplot(3,3,3);
plot(D1_L_sym_eigval);
title('D1 Lsym eigenvalues');
subplot(3,3,4);
plot(D2_L_unnorm_eigval);
title('D2 Lunnorm eigenvalues');
subplot(3,3,5);
plot(D2_L_rw_eigval);
title('D2 Lrw eigenvalues');
subplot(3,3,6);
plot(D2_L_sym_eigval);
title('D2 Lsym eigenvalues');


% find the smallest eigenvectors for D1_L_sym
[D1_L_sym_eigvect_k2 val1] =  eigs(D1_L_sym, 2,'sm');
[D1_L_sym_eigvect_k3 val2] =  eigs(D1_L_sym, 3,'sm');
[D1_L_sym_eigvect_k4 val3] =  eigs(D1_L_sym, 4,'sm');

% V matrix of eigenvectors

% find the smallest eigenvectors for D2_L_sym
[D2_L_sym_eigvect_k2 val4] =  eigs(D2_L_sym, 2,'sm');
[D2_L_sym_eigvect_k3 val5] =  eigs(D2_L_sym, 3,'sm');
[D2_L_sym_eigvect_k4 val6] =  eigs(D2_L_sym, 4,'sm');

D1_norm_k2 = sqrt(sum(D1_L_sym_eigvect_k2.^2,2));
D1_norm_k3 = sqrt(sum(D1_L_sym_eigvect_k3.^2,2));
D1_norm_k4 = sqrt(sum(D1_L_sym_eigvect_k4.^2,2));

D2_norm_k2 = sqrt(sum(D2_L_sym_eigvect_k2.^2,2));
D2_norm_k3 = sqrt(sum(D2_L_sym_eigvect_k3.^2,2));
D2_norm_k4 = sqrt(sum(D2_L_sym_eigvect_k4.^2,2));

% normalize V
for i = 1:length(D2_L_sym_eigvect_k2)
    
    D1_Lsym_k2_norm(i,:) = D1_L_sym_eigvect_k2(i,:)/ D1_norm_k2(i);
    D1_Lsym_k3_norm(i,:) = D1_L_sym_eigvect_k3(i,:)/ D1_norm_k3(i);
    D1_Lsym_k4_norm(i,:) = D1_L_sym_eigvect_k4(i,:)/ D1_norm_k4(i);

    D2_L_sym_k2_norm(i,:) = D2_L_sym_eigvect_k2(i,:)/ D2_norm_k2(i);
    D2_L_sym_k3_norm(i,:) = D2_L_sym_eigvect_k3(i,:)/ D2_norm_k3(i);
    D2_L_sym_k4_norm(i,:) = D2_L_sym_eigvect_k4(i,:)/ D2_norm_k4(i);
    
end

% [idx,C] = kmeans(D1_Lsym_k2_norm,2);
% 
% figure;
% plot(D1(idx==1,1),D1(idx==1,2),'r.','MarkerSize',12)
% hold on
% plot(D1(idx==2,1),D1(idx==2,2),'b.','MarkerSize',12)
% plot(C(:,1),C(:,2),'kx',...
%      'MarkerSize',15,'LineWidth',3)
 
