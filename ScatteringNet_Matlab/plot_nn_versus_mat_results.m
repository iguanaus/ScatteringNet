wgts = cell(0);
bias = cell(0);
for i=0:4
    wgts{i+1} = transpose(load('spectrums/Dielectric_Corrected_TiO2/w_'+string(i)+'.txt'));
    bias{i+1} = load('spectrums/Dielectric_Corrected_TiO2/b_'+string(i)+'.txt');
end

input = [39.4;49.1;66.3;56.5;47.4]
lambda = linspace(400, 801, 401)';
result = NN(wgts,bias,input)
result2 = scatter_0_generate_spectrum(input);
result2 = result2(1:2:401,1);
hold on
plot(result)
plot(result2)
legend('NN','RealResult');
hold off

