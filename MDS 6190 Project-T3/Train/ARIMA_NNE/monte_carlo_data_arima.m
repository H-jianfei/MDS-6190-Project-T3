%% settings
p = 2; % AR的阶数
q = 1; % MA的阶数
n = 100; % 时间序列的长度

L = 1000; % 训练和验证样本的数量

input = cell(L,1);
label = cell(L,1);

for l = 1:L
    % 随机生成AR和MA参数
    beta = unifrnd(-1, 1, [1, p]);
    theta = unifrnd(-1, 1, [1, q]);
    
    % 生成ARIMA(p,q)时间序列数据
    y = model_arima(beta, theta, p, q, n);
    
    % 计算矩
    input{l} = moments_arima(y, p);
    % 将AR和MA参数合并为一个标签，确保它们都是列向量
    label{l} = [beta(:); theta(:)]; % 将beta和theta转换为列向量后垂直串联
end

input = cell2mat(input);
label = cell2mat(label);

%% 训练-验证拆分
L_train = floor(L*0.8); % 训练样本数量
input_train = input(1:L_train,:);
label_train = label(1:L_train,:);
input_val = input(L_train+1:L,:);
label_val = label(L_train+1:L,:);

%% 保存训练和验证数据
save('nne_training_arima.mat', 'input_train', 'label_train', 'input_val', 'label_val');