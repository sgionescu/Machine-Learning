
% Problem 8_1b

clear; clc; close all;

input = load('BostonListing.mat');
latitude = input.latitude;
longitude = input.longitude;
neighbourhood = input.neighbourhood;

k = 3;
points_per_cluster = [500,500,500];
[D1, D1_label] = sample_circle(k, points_per_cluster );
[D2, D2_label] = sample_spiral(k, points_per_cluster );


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
figure(1)
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

% find the V matrix for L, k = 3
[D1_V_L_k3 L3_val1] =  eigs(D1_L_unnorm, 3,'sm');
[D2_V_L_k3 L3_val2] =  eigs(D2_L_unnorm, 3,'sm');


% find the V matrix for Lrw, k = 3
[D1_V_Lrw_k3 Lrw3_val1] =  eigs(D1_L_rw, 3,'sm');
[D2_V_Lrw_k3 Lrw3_val2] =  eigs(D2_L_rw, 3,'sm');

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

    D2_Lsym_k2_norm(i,:) = D2_L_sym_eigvect_k2(i,:)/ D2_norm_k2(i);
    D2_Lsym_k3_norm(i,:) = D2_L_sym_eigvect_k3(i,:)/ D2_norm_k3(i);
    D2_Lsym_k4_norm(i,:) = D2_L_sym_eigvect_k4(i,:)/ D2_norm_k4(i);
    
end

[idx2,C2] = kmeans(D1_Lsym_k2_norm,2);

figure(2)
subplot(2,3,1)
plot(D1(idx2==1,1),D1(idx2==1,2),'r.','MarkerSize',10)
hold on
plot(D1(idx2==2,1),D1(idx2==2,2),'b.','MarkerSize',10)
plot(C2(:,1),C2(:,2),'kx',...
     'MarkerSize', 12,'LineWidth',3)
title('D1, k = 2')

[idx3,C3] = kmeans(D1_Lsym_k3_norm,3);
subplot(2,3,2)
plot(D1(idx3==1,1),D1(idx3==1,2),'r.','MarkerSize',10)
hold on
plot(D1(idx3==2,1),D1(idx3==2,2),'b.','MarkerSize',10)
hold on
plot(D1(idx3==3,1),D1(idx3==3,2),'g.','MarkerSize',10)
plot(C3(:,1),C3(:,2),'kx',...
     'MarkerSize', 12,'LineWidth',3)
title('D1, k = 3')

[idx4,C4] = kmeans( D1_Lsym_k4_norm, 4);
subplot(2,3,3)
plot(D1(idx4==1,1),D1(idx4==1,2),'r.','MarkerSize',10)
hold on
plot(D1(idx4==2,1),D1(idx4==2,2),'b.','MarkerSize',10)
hold on
plot(D1(idx4 == 3,1),D1(idx4 == 3,2),'g.','MarkerSize',10)
hold on
plot(D1(idx4 == 4,1),D1(idx4 == 4,2),'k.','MarkerSize',10)
plot(C4(:,1),C4(:,2),'kx',...
     'MarkerSize', 12,'LineWidth',3)
title('D1, k = 4')

[D2_idx2,D2_C2] = kmeans(D2_Lsym_k2_norm,2);

subplot(2,3,4)
plot(D2(D2_idx2==1,1),D2(D2_idx2==1,2),'r.','MarkerSize',10)
hold on
plot(D2(D2_idx2==2,1),D2(D2_idx2==2,2),'b.','MarkerSize',10)
plot(D2_C2(:,1), D2_C2(:,2),'kx',...
     'MarkerSize', 12,'LineWidth',3)
title('D2, k = 2')

[D2_idx3,D2_C3] = kmeans(D2_Lsym_k3_norm,3);
subplot(2,3,5)
plot(D2(D2_idx3==1,1),D2(D2_idx3==1,2),'r.','MarkerSize',10)
hold on
plot(D2(D2_idx3==2,1),D2(D2_idx3==2,2),'b.','MarkerSize',10)
hold on
plot(D2(D2_idx3==3,1),D2(D2_idx3==3,2),'g.','MarkerSize',10)
plot(D2_C3(:,1),D2_C3(:,2),'kx',...
     'MarkerSize', 12,'LineWidth',3)
title('D2, k = 3')


[D2_idx4, D2_C4] = kmeans( D2_Lsym_k4_norm, 4);
subplot(2,3,6)
plot(D2(D2_idx4==1,1),D2(D2_idx4==1,2),'r.','MarkerSize',10)
hold on
plot(D2(D2_idx4==2,1),D2(D2_idx4==2,2),'b.','MarkerSize',10)
hold on
plot(D2(D2_idx4 == 3,1),D2(D2_idx4 == 3,2),'g.','MarkerSize',10)
hold on
plot(D2(D2_idx4 == 4,1),D2(D2_idx4 == 4,2),'k.','MarkerSize',10)
plot(D2_C4(:,1), D2_C4(:,2),'kx',...
     'MarkerSize', 12,'LineWidth',3)
title('D2, k = 4')

% kmeans for L and k = 3
[LD1_idx3,LD1_C3] = kmeans(D1_V_L_k3,3);
[LD2_idx3,LD2_C3] = kmeans(D2_V_L_k3,3);

% kmeans for Lrw and k = 3
[LrwD1_idx3,LrwD1_C3] = kmeans(D1_V_Lrw_k3, 3);
[LrwD2_idx3,LrwD2_C3] = kmeans(D2_V_Lrw_k3, 3);

% part 8b_iii

figure(3)
subplot(2,3,1)
plot3(D1_V_L_k3(LD1_idx3==1,1),D1_V_L_k3(LD1_idx3==1,2),D1_V_L_k3(LD1_idx3==1,3),'r.','MarkerSize',10)
hold on
plot3(D1_V_L_k3(LD1_idx3==2,1),D1_V_L_k3(LD1_idx3==2,2),D1_V_L_k3(LD1_idx3==2,3) ,'b.','MarkerSize',10)
hold on
plot3(D1_V_L_k3(LD1_idx3==3,1),D1_V_L_k3(LD1_idx3==3,2),D1_V_L_k3(LD1_idx3==3,3),'g.','MarkerSize',10)

title('SC-1, D1, k = 3')

subplot(2,3,2)
plot3(D1_V_Lrw_k3(LrwD1_idx3==1,1),D1_V_Lrw_k3(LrwD1_idx3==1,2),D1_V_Lrw_k3(LrwD1_idx3==1,2),'r.','MarkerSize',10)
hold on
plot3(D1_V_Lrw_k3(LrwD1_idx3==2,1),D1_V_Lrw_k3(LrwD1_idx3==2,2),D1_V_Lrw_k3(LrwD1_idx3==2,3) ,'b.','MarkerSize',10)
hold on
plot3(D1_V_Lrw_k3(LrwD1_idx3==3,1),D1_V_Lrw_k3(LrwD1_idx3==3,2),D1_V_Lrw_k3(LrwD1_idx3==3,3),'g.','MarkerSize',10)

title('SC-2, D1, k = 3')


subplot(2,3,3)
plot3(D1_Lsym_k3_norm(idx3==1,1),D1_Lsym_k3_norm(idx3==1,2),D1_Lsym_k3_norm(idx3==1,3),'r.','MarkerSize',10)
hold on
plot3(D1_Lsym_k3_norm(idx3==2,1),D1_Lsym_k3_norm(idx3==2,2),D1_Lsym_k3_norm(idx3==2,3) ,'b.','MarkerSize',10)
hold on
plot3(D1_Lsym_k3_norm(idx3==3,1),D1_Lsym_k3_norm(idx3==3,2), D1_Lsym_k3_norm(idx3==3,3),'g.','MarkerSize',10)

title('SC-3, D1, k = 3')


% plot D2

subplot(2,3,4)
plot3(D2_V_L_k3(LD2_idx3==1,1),D2_V_L_k3(LD2_idx3==1,2),D2_V_L_k3(LD2_idx3==1,3),'r.','MarkerSize',10)
hold on
plot3(D2_V_L_k3(LD2_idx3==2,1),D2_V_L_k3(LD2_idx3==2,2),D2_V_L_k3(LD2_idx3==2,3) ,'b.','MarkerSize',10)
hold on
plot3(D2_V_L_k3(LD2_idx3==3,1),D2_V_L_k3(LD2_idx3==3,2),D2_V_L_k3(LD2_idx3==3,3),'g.','MarkerSize',10)

title('SC-1, D2, k = 3')


subplot(2,3,5)
plot3(D2_V_Lrw_k3(LrwD2_idx3==1,1),D2_V_Lrw_k3(LrwD2_idx3==1,2),D2_V_Lrw_k3(LrwD2_idx3==1,3),'r.','MarkerSize',10)
hold on
plot3(D2_V_Lrw_k3(LrwD2_idx3==2,1),D2_V_Lrw_k3(LrwD2_idx3==2,2),D2_V_Lrw_k3(LrwD2_idx3==2,3) ,'b.','MarkerSize',10)
hold on
plot3(D2_V_Lrw_k3(LrwD2_idx3==3,1),D2_V_Lrw_k3(LrwD2_idx3==3,2),D2_V_Lrw_k3(LrwD2_idx3==3,3),'g.','MarkerSize',10)

title('SC-2, D2, k = 3')


subplot(2,3,6)
plot3(D2_Lsym_k3_norm(D2_idx3==1,1),D2_Lsym_k3_norm(D2_idx3==1,2),D2_Lsym_k3_norm(D2_idx3==1,3),'r.','MarkerSize',10)
hold on
plot3(D2_Lsym_k3_norm(D2_idx3==2,1),D2_Lsym_k3_norm(D2_idx3==2,2),D2_Lsym_k3_norm(D2_idx3==2,3) ,'b.','MarkerSize',10)
hold on
plot3(D2_Lsym_k3_norm(D2_idx3==3,1),D2_Lsym_k3_norm(D2_idx3==3,2),D2_Lsym_k3_norm(D2_idx3==3,3),'g.','MarkerSize',10)

title('SC-3, D2, k = 3')


