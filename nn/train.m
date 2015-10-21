clear ; close all; clc
arglist = argv();

fprintf('Loading Data ...\n')
load(arglist{1});
m = size(X, 1);

input_layer_size  = size(X,2); % 20x20 Input Images of Digits
hidden_layer_size = str2num(arglist{2});  % 25 hidden units
num_labels = str2num(arglist{3});         % 10 labels, from 1 to 10   
                          % (note that we have mapped "0" to label 10)

fprintf('\nInitializing Neural Network Parameters ...\n')
initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

fprintf('\nProgram paused. Press enter to continue.\n');
pause;


fprintf('\nTraining Neural Network ...\n')
options = optimset('MaxIter', 50);
lambda = 1;

costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);

fprintf('\nRunning Fmincg...\n')
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

fprintf('Program paused. Press enter to continue.\n');
pause;

pred = predict(Theta1, Theta2, X);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);

t_large = [Theta1(:); Theta2(:)];


fprintf('Training Finished. Press enter to save theta values.\n');
pause;
save theta_values.mat t_large


