wgts = cell(0);
bias = cell(0);
for i=0:4
    wgts{i+1} = transpose(load('spectrums/Dielectric_TiO2_5_06_20/w_'+string(i)+'.txt'));
    bias{i+1} = load('spectrums/Dielectric_TiO2_5_06_20/b_'+string(i)+'.txt');
end

input = [33.8;32.3;70;35.2;60]
lambda = linspace(400, 801, 401)';
result = NN(wgts,bias,input)
result2 = scatter_0_generate_spectrum(input);
result2 = result2(1:2:401,1);
cost = sum((result-result2).^2);
hold on
plot(result)
plot(result2)
legend('NN','RealResult');
cost
hold off

