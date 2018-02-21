wgts = cell(0);
bias = cell(0);
for i=0:4
    wgts{i+1} = transpose(load('spectrums/5_Layer_TiO2_200_layer/w_'+string(i)+'.txt'));
    bias{i+1} = load('spectrums/5_Layer_TiO2_200_layer/b_'+string(i)+'.txt');
end
r1 = round(rand*40+30,1);
r2 = round(rand*40+30,1);
r3 = round(rand*40+30,1);
r4 = round(rand*40+30,1);
r5 = round(rand*40+30,1);
r6 = round(rand*40+30,1);
r7 = round(rand*40+30,1);
r8 = round(rand*40+30,1);
r9 = round(rand*40+30,1);
r10 = round(rand*40+30,1);
input = [r1;r2;r3]
input =  [47.5;45.3;60.6;61.8;37.5;49.6;47.8;55.9]
input2 = [49.35;45;58.9;61.8;38.1;49.6;47.6;55.6]
input3 = [49.28;53.8;53.8;54.3;44.6;54.5;52.7;51.1]

lambda = linspace(400, 801, 401)';
%result2 = NN(wgts,bias,input)
result = scatter_0_generate_spectrum(input);
result = result(1:2:401,1);
result2 = scatter_0_generate_spectrum(input2);
result2 = result2(1:2:401,1);
result3 = scatter_0_generate_spectrum(input3);
result3 = result3(1:2:401,1);
%result3 = scatter_0_generate_spectrum(input3);
%result3 = result3(1:2:401,1);
%result4 = scatter_0_generate_spectrum(input4);
%result4 = result4(1:2:401,1);

length(result)
length(result2)
cost = sum((result-result2).^2);
hold on
plot(lambda(1:2:401,1),[result,result3,result2])
%plot(result2)                              Desired Spectrum 
legend('Desired         (48 45 61 62 38 50 48 56)','Numerical     (49 54 54 54 45 54 53 51)','NN              (49 45 59 62 38 50 48 56)');
xlabel('Wavelength (nm)');
ylabel('\sigma/\pi r^2');
title('NN versus Numerical Non-Linear Optimization');

cost
hold off

