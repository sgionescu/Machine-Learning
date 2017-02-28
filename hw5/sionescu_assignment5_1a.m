% Assignment5 - Part 5.1a
% Load the train and test databases
% Extract the hour, day, and police district into three vectors
% Concatenate the three vectors into a binary vector of 41 features and 
% save it.

% Silvia Ionescu
% 10-25-2016

clear; close all; clc;

Hr = 24;

%train and test datasets
train = load('data_SFcrime_train.mat');
test = load('data_SFcrime_test.mat');

train_Dates = train.Dates;
train_DayOfWeek = train.DayOfWeek;
train_PdDistrict = train.PdDistrict;
train_Category = train.Category;

% obtain unique values
train_Category_unique = unique(train_Category);
train_DaysOfWeek_unique = {'Sunday','Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'};
train_PdDistrict_unique = unique(train_PdDistrict);

% initialize empty variables
Hour = zeros(length(train_Dates), Hr);
Day = zeros(length(train_Dates), 7);
PdDistrict = zeros(length(train_Dates), 10);
Label_crime_type = zeros(length(train_Dates),39);


for i = 1: length(train_Dates) 
    % extract the hour and make a nx24 binary vector
    disp(i);
    sample_train = char(train_Dates{i});
    sample_hour = sample_train(end-4:end-3);
    hour_num(i) = str2num(sample_hour);
    if (hour_num(i) == 0)
        hour_num(i) = 24;
        Hour(i, 24) = 1;
    else  
        Hour(i, hour_num(i)) = 1;   
    end
    
    % extract the day and make a nx7 binary vector
    day_num(i) = find(strcmp(train_DayOfWeek{i},train_DaysOfWeek_unique));
    Day(i, day_num(i)) = 1;
    
    % extract the police distric and make a nx7 binary vector
    pd_distict_num(i) = find(strcmp(train_PdDistrict{i}, train_PdDistrict_unique));
    PdDistrict(i, pd_distict_num(i)) = 1;
    
    % get label
    label(i) = find(strcmp(train_Category{i},train_Category_unique));
    Label_crime_type(i,label(i)) = 1;
end

figure(1);
hist(hour_num,24);
title('Hour Histogram');
xlabel('Hour (hr)');
ylabel('Counts');

figure(2);
hist(day_num,7);
title('Day of the week histogram');
xlabel('Day of Week (day)');
ylabel('Counts');

figure(3);
hist(pd_distict_num,10);
title('Police department district histogram');
xlabel('PdDistrict (district)');
ylabel('Counts');

train_data = [Hour, Day, PdDistrict] ;

save('train_data','train_data','label');

% calculate the most likely hour for each type of crime
hour_max_per_label = zeros(1,39);
for k = 1:39
    disp('k:');
    disp(k);
    hour_per_label =  sum(Hour(find(label == k),:),1);
    find_hour = find(hour_per_label == max(hour_per_label));
    if length(find_hour) < 2
        hour_max_per_label(k) = find_hour;
    else
        hour_max_per_label(k) = find_hour(1,1);
    end
    
end

crime_per_PdDistrict = zeros(1,10);
for r = 1:10
   a = sum(Label_crime_type(find(pd_distict_num == r),:),1); 
   crime_per_PdDistrict(r) = find(a == max(a));
end

%%%%%%%%% Test preprocessing %%%%%%%%%

test_Dates = test.Dates_test;
test_DayOfWeek = test.DayOfWeek_test;
test_PdDistrict = test.PdDistrict_test;


% obtain unique values
test_DaysOfWeek_unique = {'Sunday','Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'};
test_PdDistrict_unique = unique(test_PdDistrict);

% initialize with zero
Hour_test = zeros(length(test_Dates), Hr);
Day_test = zeros(length(test_Dates), 7);
PdDistrict_test = zeros(length(test_Dates), 10);


for t = 1: length(test_Dates) 
    % hours
    disp(t);
    sample_test = char(test_Dates{t});
    sample_hour_test = sample_test(end-7:end-6);
    
    hour_num_test(t) = str2num(sample_hour_test);
    if (hour_num_test(t) == 0)
        hour_num_test(t) = 24;
        Hour_test(t, 24) = 1;
    else  
        Hour_test(t, hour_num_test(t)) = 1;   
    end
    
    % day
    day_num_test(t) = find(strcmp(test_DayOfWeek{t},test_DaysOfWeek_unique));
    Day_test(t, day_num_test(t)) = 1;
    
    % police department
    pd_distict_num_test(t) = find(strcmp(test_PdDistrict{t}, test_PdDistrict_unique));
    PdDistrict_test(t, pd_distict_num_test(t)) = 1;
 
end

test_data = [Hour_test, Day_test, PdDistrict_test] ;

save('test_data','test_data');