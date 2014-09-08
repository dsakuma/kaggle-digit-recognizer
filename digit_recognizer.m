%========== Initialization
clear ; close all; clc


%========== Load training and cross validation set
fprintf('\nLoading training and cross validation set\n')
load data.mat;

%{
%========== Cross Validation (hidden layer size)

hidden_layer_sizes = [40; 44; 48; 52; 56; 60];
%hidden_layer_sizes = [40; 42];
lambda = 0;
epochs = 200;

%Train_correct = zeros(size(hidden_layer_sizes));
%CV_correct = zeros(size(hidden_layer_sizes));
J_train = zeros(size(hidden_layer_sizes));
J_cv = zeros(size(hidden_layer_sizes));

for i = 1:length(hidden_layer_sizes)

	hidden_layer_size = hidden_layer_sizes(i);

	initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
	initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);
	initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

	fprintf('\nTraining NN\n')

	% Options
	options = optimset('MaxIter', epochs);

	% Create "short hand" for the cost function to be minimized
	costFunction = @(p) nnCostFunction(p, ...
	                                   input_layer_size, ...
	                                   hidden_layer_size, ...
	                                   num_labels, X_train, y_train, lambda);

	% Now, costFunction is a function that takes in only one argument (the
	% neural network parameters)
	[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

	% Obtain Theta1 and Theta2 back from nn_params
	Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
	                 hidden_layer_size, (input_layer_size + 1));

	Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
	                 num_labels, (hidden_layer_size + 1));


	pred_train = predict(Theta1, Theta2, X_train);
	pred_cv = predict(Theta1, Theta2, X_cv);


	%Train_correct(i) = mean(double(pred_train == y_train)) * 100;
	%CV_correct(i) = mean(double(pred_cv == y_cv)) * 100;
	[J_train(i) dummy] = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X_train, y_train, lambda);
	[J_cv(i) dummy] = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X_cv, y_cv, lambda);

endfor


plot(hidden_layer_sizes, J_train, "b");
hold on;
plot(hidden_layer_sizes, J_cv, "r");
%}


%{
%========== Cross Validation (epochs)

hidden_layer_size = 140;
lambda = 0;
epochs_arr = [100,200,300,400,500,600];
%epochs_arr = [1,2];

Train_correct = zeros(size(epochs_arr));
CV_correct = zeros(size(epochs_arr));
J_train = zeros(size(epochs_arr));
J_cv = zeros(size(epochs_arr));

% Initializing Theta
fprintf('\nInitializing parameters\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];


for i = 1:length(epochs_arr)

	epochs = epochs_arr(i);

	fprintf('\nTraining NN\n')

	% Options
	options = optimset('MaxIter', epochs);

	% Create "short hand" for the cost function to be minimized
	costFunction = @(p) nnCostFunction(p, ...
	                                   input_layer_size, ...
	                                   hidden_layer_size, ...
	                                   num_labels, X_train, y_train, lambda);

	% Now, costFunction is a function that takes in only one argument (the
	% neural network parameters)
	[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

	% Obtain Theta1 and Theta2 back from nn_params
	Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
	                 hidden_layer_size, (input_layer_size + 1));

	Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
	                 num_labels, (hidden_layer_size + 1));


	pred_train = predict(Theta1, Theta2, X_train);
	pred_cv = predict(Theta1, Theta2, X_cv);


	Train_correct(i) = mean(double(pred_train == y_train)) * 100;
	CV_correct(i) = mean(double(pred_cv == y_cv)) * 100;
	[J_train(i) dummy] = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X_train, y_train, lambda);
	[J_cv(i) dummy] = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X_cv, y_cv, lambda);


endfor


Train_correct
CV_correct

plot(epochs_arr, J_train, "b");
hold on;
plot(epochs_arr, J_cv, "r");
%}


%========== Cross Validation v2 (epochs)

hidden_layer_size = 54;
lambda = 0;
epochs = 400;

J_train = zeros(epochs,1);
J_cv = zeros(epochs,1);

% Initializing Theta
fprintf('\nInitializing parameters\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

nn_params = initial_nn_params;
for i = 1:epochs

	fprintf('\nTraining NN - epoch: %d\n', i)

	% Options
	options = optimset('MaxIter', 1);

	% Create "short hand" for the cost function to be minimized
	costFunction = @(p) nnCostFunction(p, ...
	                                   input_layer_size, ...
	                                   hidden_layer_size, ...
	                                   num_labels, X_train, y_train, lambda);

	% Now, costFunction is a function that takes in only one argument (the
	% neural network parameters)
	[nn_params, cost] = fmincg(costFunction, nn_params, options);

	% Obtain J
	J_train(i) = cost;
	[J_cv(i) dummy] = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X_cv, y_cv, lambda);


endfor

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));


%plot
plot(1:1:epochs, J_train, "b");
hold on;
plot(1:1:epochs, J_cv, "r");

J_train
J_cv




%{
%========== Cross Validation (lambda)

hidden_layer_size = 140;
epochs = 200;
lambdas = [0, 0.01, 0.05, 0.1, 0.5, 1, 5, 10];


%Train_correct = zeros(size(lambdas));
%CV_correct = zeros(size(lambdas));
J_train = zeros(size(lambdas));
J_cv = zeros(size(lambdas));

% Initializing Theta
fprintf('\nInitializing parameters\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];


for i = 1:length(lambdas)

	lambda = lambdas(i);

	fprintf('\nTraining NN\n')

	% Options
	options = optimset('MaxIter', epochs);

	% Create "short hand" for the cost function to be minimized
	costFunction = @(p) nnCostFunction(p, ...
	                                   input_layer_size, ...
	                                   hidden_layer_size, ...
	                                   num_labels, X_train, y_train, lambda);

	% Now, costFunction is a function that takes in only one argument (the
	% neural network parameters)
	[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

	% Obtain Theta1 and Theta2 back from nn_params
	Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
	                 hidden_layer_size, (input_layer_size + 1));

	Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
	                 num_labels, (hidden_layer_size + 1));


	pred_train = predict(Theta1, Theta2, X_train);
	pred_cv = predict(Theta1, Theta2, X_cv);


	%Train_correct(i) = mean(double(pred_train == y_train)) * 100;
	%CV_correct(i) = mean(double(pred_cv == y_cv)) * 100;
	[J_train(i) dummy] = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X_train, y_train, lambda);
	[J_cv(i) dummy] = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X_cv, y_cv, lambda);


endfor

lambdas
J_train
J_cv

plot(lambdas, J_train, "b");
hold on;
plot(lambdas, J_cv, "r");
%}

%{
%========== Custom training

hidden_layer_size = 140;
epochs = 200;
lambda = 0;


% Initializing Theta
fprintf('\nInitializing parameters\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];


fprintf('\nTraining NN\n')

% Options
options = optimset('MaxIter', epochs);

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X_train, y_train, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));


pred_train = predict(Theta1, Theta2, X_train);
pred_cv = predict(Theta1, Theta2, X_cv);


Train_correct = mean(double(pred_train == y_train)) * 100;
CV_correct = mean(double(pred_cv == y_cv)) * 100;
[J_train dummy] = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X_train, y_train, lambda);
[J_cv dummy] = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X_cv, y_cv, lambda);


% Show results
Train_correct
CV_correct
J_train
J_cv
%}



%========== Test Evaluation
fprintf('\nPredicting\n')

TestSet = dlmread('test.csv',',');


% reducing dimension
idx_zeroed = idx_zeroed .- 1;
TestSet(:, idx_zeroed) = [];


X_test = TestSet(2:end, :);

pred_test = predict(Theta1, Theta2, X_test);

pred_test(pred_test == 10) = 0;
save file.txt pred_test;

%save csv and txt








%{
	pred_train = predict(Theta1, Theta2, X_train);
	pred_cv = predict(Theta1, Theta2, X_cv);


	Train_correct(i) = mean(double(pred_train == y_train)) * 100;
	CV_correct(i) = mean(double(pred_cv == y_cv)) * 100;


endfor

% Ploting
lambdas
Train_correct
CV_correct

plot(lambdas, Train_correct, "b");
hold on;
plot(lambdas, CV_correct, "r");
%}
