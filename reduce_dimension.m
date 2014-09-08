%========== Initialization
clear ; close all; clc


TrainSet = dlmread('train.csv',',');


% Removing zeroed features
sum_features = sum(TrainSet(2:end,:));
idx_zeroed = find(sum_features < 2000);
TrainSet(:, idx_zeroed) = [];
fprintf('\nNumber of removed dimensions: %d\n', size(idx_zeroed));

% Normalizing
TrainSet(2:end, 2:end) = TrainSet(2:end, 2:end) ./ 255;

% Partitioning into Test and CV Sets
X_train = TrainSet(2:29401, 2:end); %29400
X_cv = TrainSet(29402:end, 2:end); %12600
y_train = TrainSet(2:29401, 1);
y_cv = TrainSet(29402:end, 1);

% Changing label 0 to 10
y_train(y_train == 0) = 10;
y_cv(y_cv == 0) = 10;

input_layer_size = size(X_train,2);
num_labels = 10;
m_train = size(y_train);
m_cv = size(y_cv);

% Saving sets
save data.mat X_train X_cv y_train y_cv idx_zeroed num_labels input_layer_size m_train m_cv

fprintf('\nFinished.')