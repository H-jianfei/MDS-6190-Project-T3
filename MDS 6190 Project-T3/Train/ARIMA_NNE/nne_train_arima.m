%% nne_train_arima.m

%% 加载训练和验证样本
load('nne_training_arima.mat')

%% 数据预处理
% 归一化输入数据 因此在训练时出现NaN,我想应该是因为有极值出现
[~, mu, sigma] = zscore(input_train); % 计算训练集的均值和标准差，并标准化
input_train_norm = (input_train - mu) ./ sigma; % 标准化训练数据
input_val_norm = (input_val - mu) ./ sigma; % 使用相同的均值和标准差标准化验证数据

%% 训练神经网络
num_nodes = 32; % 隐藏节点数
dim_input = size(input_train, 2); % 输入维度
num_params = size(label_train, 2); % 参数数量，确保标签的列数与参数数量匹配

% 定义训练选项
% 这是复制的AR(1)的，我使用的做出了一点修改
% opts = trainingOptions( 'adam', ...
%                         'L2Regularization', 0, ...
%                         'ExecutionEnvironment', 'cpu', ...
%                         'MaxEpochs', 500, ...
%                         'InitialLearnRate', 0.01, ...
%                         'GradientThreshold', 1, ...
%                         'MiniBatchSize', 500, ...
%                         'Plots','none', ...
%                         'Verbose', true, ...
%                         'VerboseFrequency', 100, ...
%                         'ValidationData', {input_val_norm, label_val},...
%                         'ValidationFrequency', 100);
opts = trainingOptions('adam', ...
    'L2Regularization', 1e-4, ...
    'ExecutionEnvironment', 'cpu', ...
    'MaxEpochs', 500, ...
    'InitialLearnRate', 0.001, ...
    'GradientThreshold', 1, ...
    'MiniBatchSize', 500, ...
    'Plots','training-progress', ...
    'Verbose', true, ...
    'VerboseFrequency', 100, ...
    'ValidationData', {input_val_norm, label_val},...
    'ValidationFrequency', 100);

layers = [  featureInputLayer(dim_input)
            fullyConnectedLayer(num_nodes)
            reluLayer
            fullyConnectedLayer(num_params) % 输出层维度与参数数量相匹配
            regressionLayer
            ];

[net, info] = trainNetwork(input_train_norm, label_train, layers, opts);

disp("Final validation loss is: " + info.FinalValidationLoss)

%% 显示验证集上的估计与真实值
pred_val = predict(net, input_val);

% 绘制估计值与真实值
index = abs(pred_val) > 1;
pred_val_index = pred_val;
pred_val_index(index) = 0;

figure('position', [750,500,250,250])
sgtitle('Estimate vs. Truth in Validation')
subplot(2, 2, 1)
scatter(label_val(:,1), pred_val_index(:,1), '.')
xlabel('Beta1')
ylabel('Estimated Beta1')
subplot(2, 2, 2)
scatter(label_val(:,2), pred_val_index(:,2), '.')
xlabel('Beta2')
ylabel('Estimated Beta2')
subplot(2, 2, 3)
scatter(label_val(:,3), pred_val_index(:,3), '.')
xlabel('Theta1')
ylabel('Estimated Theta1')
subplot(2, 2, 4)
scatter(label_val(:,4), pred_val_index(:,4), '.')
xlabel('Theta2')
ylabel('Estimated Theta2')