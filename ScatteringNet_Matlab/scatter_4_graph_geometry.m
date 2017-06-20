data = load('data.mat');
lambda = linspace(400, 800, 401)';
omega = 2*pi./lambda;

values = [];
myspects = [];
%33.8_32.3_36.3_35.2_38.9
r1_1 = 33.8;
r2_1 = 32.3;
r3_1 = 36.3;
r4_1 = 35.2;
r5_1 = 38.9;
myname_1 = num2str(strcat('Desired :',num2str(r1_1),'-',num2str(r2_1),'-',num2str(r3_1),'-',num2str(r4_1),'-',num2str(r5_1)));
spect1 = scatter_0_generate_spectrum([r1_1,r2_1,r3_1,r4_1,r5_1])
r1_2 = 39.3;
r2_2 = 30.0;
r3_2 = 31.35;
r4_2 = 37.1;
r5_2 = 32.8;
myname_2 = num2str(strcat('NN :',num2str(r1_2),'-',num2str(r2_2),'-',num2str(r3_2),'-',num2str(r4_2),'-',num2str(r5_2)));
spect2 = scatter_0_generate_spectrum([r1_2,r2_2,r3_2,r4_2,r5_2])
r1_3 = 33.8;
r2_3 = 32.3;
r3_3 = 36.3;
r4_3 = 35.2;
r5_3 = 38.9;
myname_3 = num2str(strcat('MatLab :',num2str(r1_3),'-',num2str(r2_3),'-',num2str(r3_3),'-',num2str(r4_3),'-',num2str(r5_3)));
spect3 = scatter_0_generate_spectrum([r1_3,r2_3,r3_3,r4_3,r5_3])
hold on
%plot(lambda,[spect1(1:1:end,1)-spect2(1:1:end,1),spect1(1:1:end,1)-spect3(1:1:end,1)]);%spect(1:5:501,1));
plot(lambda,[spect1(1:1:end,1),spect2(1:1:end,1),spect3(1:1:end,1)]);%spect(1:5:501,1));
%plot(lambda,spect1(1:1:end,1));
xlabel('Wavelength (nm)');
ylabel('\sigma/\pi r^2');
%title('Residuals');
title('NN versus Matlab Non-Linear Optimization');
%legend('14/25/24/38 after 2 hours','14/25/24/38 after 6 hours');
legend(myname_1,myname_2,myname_3);
%legend('Desired for 49/23/22/12/21','NN begin training','NN for reveresed 48/21/18/10/27');

