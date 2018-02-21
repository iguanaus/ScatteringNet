wgts = cell(0);
bias = cell(0);
for i=0:4
    wgts{i+1} = transpose(load('spectrums/J-Aggregate/w_'+string(i)+'.txt'));
    bias{i+1} = load('spectrums/J-Aggregate/b_'+string(i)+'.txt');
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
input = [56.8;58.4;45.5]
val = 500;
nn_input = [56.8;58.4;45.5;500]
%input =  [40;53;66;35;35]
%input2 = [60;45;67;30;35]
%input3 = [30;60;50;45;65]
%input4 = [40;60;55;45;70]
%input = [56.8;58.4;45.5;33.8;69.8;44;58;55.2]
%input = [50.0;31.1;40.3;30.3;45.6;60.4;30.2;56.5;45.4;60.5]
lambda = linspace(400, 801, 401)';
result  =  scatter_0_generate_spectrum_jagg(input,val)
result = result(1:2:401,1);
result2 = NN(wgts,bias,nn_input)

%result2 = scatter_0_generate_spectrum(input2);
%result2 = result2(1:2:401,1);
%result3 = scatter_0_generate_spectrum(input3);
%result3 = result3(1:2:401,1);
%result4 = scatter_0_generate_spectrum(input4);
%result4 = result4(1:2:401,1);

length(result)
length(result2)
cost = sum((result-result2).^2);
hold on
plot(lambda(1:2:401,1),result,lambda(1:2:401,1),result2,'-.')
%plot(lambda(1:2:399,1),result,lambda(1:2:399,1),result2,'-.',lambda(1:2:399,1),result_ex,':',lambda(1:2:399,1),result_ex2,'--')
%plot(result2)                              Desired Spectrum 
legend('Simulation','NN Approx');
xlabel('Wavelength (nm)');
ylabel('\sigma/\pi r^2');
title('Inverse Design for J-Aggregate');
set(gca,'fontsize',16)
set(findall(gca, 'Type', 'Line'),'LineWidth',2);
hold off
%axes('Position',[.2 .7 .2 .2])
%box on
%set(gca,'fontsize',16)
%set(findall(gca, 'Type', 'Line'),'LineWidth',2);

%plot(lambda(1:2:100,1),result(1:1:50,1),lambda(1:2:100,1),result2(1:1:50,1),'-.',lambda(1:2:100,1),result_ex(1:1:50,1),':',lambda(1:2:100,1),result_ex2(1:1:50,1),'--')
%set(gca,'fontsize',14)
%xlim([445 485])
%ylim([2.4 3.7])
%set(findall(gca, 'Type', 'Line'),'LineWidth',2);


