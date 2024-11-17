clear;

%% Load Trained Network
load('trained_net.mat', 'net');

%% Load and Prepare Your Real Data
% Load your training data
your_train_data = readmatrix('train1.xlsx');
your_train_data = your_train_data(:);

% Compute moments for your training data
moments_your_train = moments2(your_train_data);

% Predict beta coefficients using the trained network
estimated_beta_train = predict(net, moments_your_train);

% Display estimated beta coefficients
disp('Estimated beta coefficients for your training data:');
disp(estimated_beta_train);

% Load your validation data
your_val_data = readmatrix('test1.xlsx');
your_val_data = your_val_data(:);

% Compute moments for your validation data
moments_your_val = moments2(your_val_data);

% Predict beta coefficients using the trained network
estimated_beta_val = predict(net, moments_your_val);

% Display estimated beta coefficients
disp('Estimated beta coefficients for your validation data:');
disp(estimated_beta_val);
