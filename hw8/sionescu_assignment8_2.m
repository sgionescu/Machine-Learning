
% Problem 2

clear; clc; close all;
input = load('BostonListing.mat');

latitude = input.latitude;
longitude = input.longitude;
nbh = input.neighbourhood;

position = [latitude, longitude];

sigma = 0.01;
distance = pdist2(position, position, 'euclidean');
similarity_matrix = exp(-((distance.^2)/(2*sigma^2))); 
W = similarity_matrix;
degree_matrix = sum(W, 2);
degree_matrix = diag(degree_matrix);

% D1 -calculate Laplacians
L_unnorm = degree_matrix - W;
L_sym = (degree_matrix^-(1/2)) * L_unnorm * (degree_matrix^-(1/2));

for k = 1:25
    [Lsym_eigvect val] =  eigs(L_sym, k,'sm');
    
    Lsym_eigvect_norm = normr(Lsym_eigvect);
    [idx, C] = kmeans(Lsym_eigvect_norm, k);
    if k == 5
        idx5 = idx;
    end
    sum = 0;
    for j = 1:k
        neighbourhood = categorical(nbh(find(idx == j)));
        num = countcats(neighbourhood);
        max_ni = max(num);
        sum = sum + max_ni/2558;
    end
    purity(k) = sum;
end

figure(1)
plot(1:25, purity);
title('Purity metric vs. k ')
ylabel('Purity');
xlabel('k values');

% Hw8_2b

figure(2)
plot(longitude(idx5==1),latitude(idx5==1),'r.','MarkerSize',10)
hold on
plot(longitude(idx5==2),latitude(idx5==2),'b.','MarkerSize',10)
hold on
plot(longitude(idx5==3),latitude(idx5==3),'g.','MarkerSize',10)
hold on
plot(longitude(idx5==4),latitude(idx5==4),'m.','MarkerSize',10)
hold on
plot(longitude(idx5==5),latitude(idx5==5),'y.','MarkerSize',10)
plot_google_map





