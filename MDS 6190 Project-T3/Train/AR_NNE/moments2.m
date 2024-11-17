function output = moments2(returns)
    % Computes statistical moments
    returns = returns(:);
    p = 6; % Maximum lag
    
    maxLag = p;
    
    % Compute moments
    mean_return = mean(returns);
    variance_return = var(returns);
    skewness_return = skewness(returns);
    kurtosis_return = kurtosis(returns);
    
    % Compute autocovariances and autocorrelations
    autocov_values = zeros(1, maxLag);
    acf_values = zeros(1, maxLag);
    for k = 1:maxLag
        lagged_returns = returns(1:end - k);
        current_returns = returns(k + 1:end);
        
        C = cov(current_returns, lagged_returns, 1);
        autocov_values(k) = C(1, 2);
        acf_values(k) = C(1, 2) / sqrt(C(1,1) * C(2,2));
        
        % Handle NaN or Inf values
        if isnan(autocov_values(k)) || isinf(autocov_values(k))
            autocov_values(k) = 0;
        end
        if isnan(acf_values(k)) || isinf(acf_values(k))
            acf_values(k) = 0;
        end
    end
    
    % Aggregate moments
    output = [mean_return, variance_return, skewness_return, kurtosis_return, autocov_values, acf_values];
end
nne_gen