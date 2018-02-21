wgts = cell(0);
bias = cell(0);
for i=0:4
    wgts{i+1} = transpose(load('spectrums/3_Layer_TiO2_Final/w_'+string(i)+'.txt'));
    bias{i+1} = load('spectrums/3_Layer_TiO2_Final/b_'+string(i)+'.txt');
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
%input = [r1;r2;r3]
input =  [31.2;37.5;67.7] %Desired 31.2_37.5_67.7
input2 = [50.0;50.0;50] %One
input3 = [41.1;31.8;33.9] %Two
input4 = [39.7;30;34.8] %Three
input5 = [31.9;37;67.6] %Four



%input3 = [30;60;50;45;65]
%input4 = [40;60;55;45;70]
%input = [50.0;31.1;40.3;30.3;45.6;60.4;30.2;56.5;45.4;60.5]
lambda = linspace(400, 801, 401)';
%result2 = NN(wgts,bias,input)
result = scatter_0_generate_spectrum(input);
result = result(1:2:401,1);
result2 = scatter_0_generate_spectrum(input2);
result2 = result2(1:2:401,1);
result3 = scatter_0_generate_spectrum(input3);
result3 = result3(1:2:401,1);
result4 = scatter_0_generate_spectrum(input4);
result4 = result4(1:2:401,1);
result5 = scatter_0_generate_spectrum(input5);
result5 = result5(1:2:401,1);
%result3 = scatter_0_generate_spectrum(input3);
%result3 = result3(1:2:401,1);
%result4 = scatter_0_generate_spectrum(input4);
%result4 = result4(1:2:401,1);

length(result)
length(result2)
cost = sum((result-result2).^2);
hold on
%plot(lambda(1:2:401,1),result)
%plot(lambda(1:2:401,1),result,lambda(1:2:401,1),result2,'.')
%plot(lambda(1:2:401,1),result,lambda(1:2:401,1),result2,'.',lambda(1:2:401,1),result3,'.')
%plot(lambda(1:2:401,1),result,lambda(1:2:401,1),result2,'.',lambda(1:2:401,1),result3,'.',lambda(1:2:401,1),result4,'.')
plot(lambda(1:2:401,1),result,lambda(1:2:401,1),result2,'.',lambda(1:2:401,1),result3,'.',lambda(1:2:401,1),result4,'.')
plot(lambda(1:2:401,1),result5,'.','MarkerSize',12)
%plot(lambda(1:2:401,1),result)
%plot(result2)                              Desired Spectrum 
legend('Desired Spectrum','NN 50 50 50','NN 41 32 34', 'NN 40 30 35','NN 32 37 68');%,'NN Result            (44.3 63.0 53.2)');
xlabel('Wavelength (nm)');
ylabel('\sigma/\pi r^2');
title('Inverse Design Results Comparison');
xlim([400 801])
ylim([0 3])
set(gca,'fontsize',12)
set(findall(gca, 'Type', 'Line'),'LineWidth',2);
%set(findall(gca, 'Type', 'Marker'),'LineWidth',20);


cost
hold off

