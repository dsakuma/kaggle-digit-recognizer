%========== Initialization
clear ; close all; clc

%% Setup the parameters you will use for this exercise
input_layer_size  = 784;  % 28x28 Input Images of Digits
hidden_layer_size = 25;   % 25 hidden units
num_labels = 10;          % 10 labels, from 1 to 10  

%========== Load training set
fprintf('\nLoad training set\n')
%load train.mat;

TrainSet = dlmread('train.csv',',');
X = TrainSet(2:end, 2:end);
y = TrainSet(2:end, 1);

y(y == 0) = 10;

m = size(y);

%========== Initialize parameters
fprintf('\nInitialize parameters\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

%unroll the parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];


%========== Training NN
fprintf('\nTraining NN\n')

options = optimset('MaxIter', 50);

lambda = 0.3;

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

fprintf('\nProgram paused. Press enter to continue.\n');
pause;


%========== Predicting
fprintf('\nPredicting\n')

pred = predict(Theta1, Theta2, X);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100) 
