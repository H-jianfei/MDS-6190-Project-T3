function y = model(beta)
    % AR(6) model function
    n = 100; % Number of observations
    p = 6;   % AR model order
    
    if length(beta) ~= p
        error('beta must be a vector of length %d.', p);
    end
    
    epsilon = randn(n,1); % Error terms
    y = zeros(n,1);       % Initialize y
    
    % Check stationarity condition
    if sum(beta.^2) >= 1
        error('Sum of squares of beta must be less than 1 for stationarity.');
    end
    
    % Initialize first p values
    y(1:p) = epsilon(1:p);
    
    % Generate AR(6) time series
    for i = p+1:n
        y(i) = beta * flip(y(i-p:i-1)) + epsilon(i);
    end
end
