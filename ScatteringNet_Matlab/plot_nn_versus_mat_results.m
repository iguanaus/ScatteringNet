wgts = cell(0);
bias = cell(0);
for i=0:4
    wgts{i+1} = transpose(load('spectrums/4_Layer_TiO2_125_layer/w_'+string(i)+'.txt'));
    bias{i+1} = load('spectrums/4_Layer_TiO2_125_layer/b_'+string(i)+'.txt');
end

input = [42.4;51.1;36.6;54.1]
lambda = linspace(400, 801, 401)';
result = NN(wgts,bias,input)
result2 = scatter_0_generate_spectrum(input);
result2 = result2(1:2:401,1);
length(result)
length(result2)
cost = sum((result-result2).^2);
hold on
plot(result)
plot(result2)
legend('NN','RealResult');
cost
hold off

