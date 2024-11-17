clear

%% Settings
p = 6; % AR model order
L = 1000; % Number of samples
lb = zeros(1, p); % Lower bounds for beta
ub = 0.9 * ones(1, p); % Upper bounds for beta

input = cell(L,1);
label = cell(L,1);
l = 1;
while l <= L
    beta_valid = false;
    while ~beta_valid
        beta = lb + (ub - lb) .* rand(1, p);
        if sum(beta.^2) <= 0.9
            beta_valid = true;
        end
    end
    
    y = model(beta);
    moments = moments2(y);
    
    if any(isnan(moments)) || any(isinf(moments))
        continue;
    end
    
    input{l} = moments;
    label{l} = beta;
    l = l + 1;
end

input = cell2mat(input);
label = cell2mat(label);

%% Training and Validation Split
L_train = floor(L * 0.8);
input_train = input(1:L_train, :);
label_train = label(1:L_train, :);
input_val = input(L_train+1:end, :);
label_val = label(L_train+1:end, :);

%% Save Data
save('nne_training.mat', 'input_train', 'label_train', 'input_val', 'label_val');
