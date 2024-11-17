clear;

%% Settings
num_nodes = 32; % Number of hidden nodes

%% Load Training and Validation Data
load('nne_training.mat');

dim_input = size(input_train, 2);
dim_output = size(label_train, 2);

%% Define Neural Network Structure
layers = [
    featureInputLayer(dim_input)
    fullyConnectedLayer(num_nodes)
    reluLayer
    fullyConnectedLayer(dim_output)
    regressionLayer
];

%% Training Options
opts = trainingOptions('adam', ...
    'L2Regularization', 0.001, ...
    'ExecutionEnvironment', 'cpu', ...
    'MaxEpochs', 500, ...
    'InitialLearnRate', 1e-4, ...
    'GradientThreshold', 10, ...
    'MiniBatchSize', 32, ...
    'Plots', 'training-progress', ...
    'Verbose', true, ...
    'VerboseFrequency', 100, ...
    'ValidationData', {input_val, label_val}, ...
    'ValidationFrequency', 100);

%% Train Neural Network
[net, info] = trainNetwork(input_train, label_train, layers, opts);

disp("Final validation loss is: " + info.FinalValidationLoss);

%% Apply Neural Network to Validation Data
pred_val = predict(net, input_val);

%% Display Results
disp('Predicted beta coefficients on validation data:');
disp(pred_val);

disp('True beta coefficients on validation data:');
disp(label_val);

%% Save Trained Network
save('trained_net.mat', 'net');
