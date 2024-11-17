function y = model_arima(beta, theta, p, q, n)
% model_arima.m - Generate ARIMA(p,q) time series data
% beta: 自回归系数，长度为 p
% theta: 移动平均系数，长度为 q
% p: AR的阶数
% q: MA的阶数
% n: 时间序列的长度

epsilon = randn(n,1); % 误差项
y = zeros(n,1); % 初始化时间序列为零

% 初始化前 p 个值
for i = 1:p
    if i == 1
        y(i) = epsilon(i); % 根据平稳性条件初始化
    else
        y(i) = 0; % 设置初始值为0
    end
end

% 生成时间序列数据

for t = p+1:n
    AR_sum = 0;
    for j = 1:p
        AR_sum = AR_sum + beta(j) * y(t-j);
    end
    MA_sum = 0;
    for j = 1:q
        MA_sum = MA_sum + theta(j) * epsilon(t-j);
    end
    y(t) = AR_sum + MA_sum + epsilon(t);
end
end