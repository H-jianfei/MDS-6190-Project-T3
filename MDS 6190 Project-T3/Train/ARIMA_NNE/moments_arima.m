function output = moments_arima(returns)
    %{
    该函数用于计算股票收益率的各种统计矩，包括自相关、自协方差等。
    输入:
        returns - 股票收益率序列
    输出:
        output - 包含各类统计矩的向量
    %}
    
    % 设置滞后阶数（可以根据需要调整）
    maxLag = 2;
    
    % 预分配矩阵
    acf_values = zeros(1, maxLag);
    autocov_values = zeros(1, maxLag);
    
    % 计算均值、方差、偏度、峰度
    mean_return = mean(returns);
    variance_return = var(returns);
    skewness_return = skewness(returns);
    kurtosis_return = kurtosis(returns);
    
    % % 计算自相关和自协方差
    for k = 1:maxLag
        % 滞后值
        lagged_returns = lagmatrix(returns, k);
        lagged_returns(isnan(lagged_returns)) = 0;

        % 自协方差
        cov_index = cov(returns, lagged_returns);
        autocov_values(1,k) = cov_index(1,2);

        % 自相关
        acf_values(1,k) = corr(returns, lagged_returns);
    end
    
    % y_cum = cumsum(returns);
    % m_cum = mean(y_cum);


    % 汇总所有矩
    % output = [mean_return, variance_return, skewness_return, kurtosis_return, acf_values, acf_values, m_cum];
    output = [mean_return, variance_return, skewness_return, kurtosis_return, acf_values, acf_values];
end
