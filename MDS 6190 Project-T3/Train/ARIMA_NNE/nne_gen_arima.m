%% settings
p = 2; % AR的阶数
q = 2; % MA的阶数
n = 100; % 时间序列的长度

L = 1000; % 训练和验证样本的数量

input_arima = cell(L,1); % 保存4个矩
label = cell(L,1);
y_data = cell(L,1);

lb_ar = -0.1; % lower bound of the AR parameter
ub_ar = 0.1; % upper bound of the AR parameter
lb_ma = 0; % lower bound of the AR parameter
ub_ma = 1; % upper bound of the AR parameter

for l = 1:L
    % 随机生成AR和MA参数
    beta = unifrnd(lb_ar, ub_ar, [1, p]);
    theta = unifrnd(lb_ma, ub_ma, [1, q]);

    % 生成ARIMA(p,q)时间序列数据
    y = model_arima(beta, theta, p, q, n);
    
    % 计算矩
    input_arima{l} = moments_arima(y);
    % 将AR和MA参数合并为一个标签，确保它们都是列向量
    label{l} = [beta(:,1), beta(:,2), theta(:,1), theta(:,2)]; % 将beta和theta转换为列向量后水平串联
end

input_arima = cell2mat(input_arima);
label = cell2mat(label);

%% 训练-验证拆分
L_train = floor(L*0.8); % 训练样本数量

input_train = input_arima(1:L_train,:);
label_train = label(1:L_train,:);

input_val = input_arima(L_train+1:L,:);
label_val = label(L_train+1:L,:);

%% 保存训练和验证数据
save('nne_training_arima.mat', 'input_train', 'label_train', 'input_val', 'label_val');