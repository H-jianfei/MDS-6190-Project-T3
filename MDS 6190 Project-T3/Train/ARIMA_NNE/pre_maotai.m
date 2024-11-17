pre_raw1 = importdata('train_1.csv').data;
pre_raw1 = pre_raw1(2:end,4);
pre_raw2 = importdata('train_2.csv').data;
pre_raw2 = pre_raw2(2:end,4);
pre_raw3 = importdata('train_3.csv').data;
pre_raw3 = pre_raw3(2:end,4);
pre_raw4 = importdata('train_4.csv').data;
pre_raw4 = pre_raw4(2:end,4);

pre_input1 = moments_arima(pre_raw1);
pred_theta1 = predict(net, pre_input1);

pre_input2 = moments_arima(pre_raw2);
pred_theta2 = predict(net, pre_input2);

pre_input3 = moments_arima(pre_raw3);
pred_theta3 = predict(net, pre_input3);

pre_input4 = moments_arima(pre_raw4);
pred_theta4 = predict(net, pre_input4);

disp(pred_theta1); 
disp(pred_theta2); 
disp(pred_theta3); 
disp(pred_theta4); 