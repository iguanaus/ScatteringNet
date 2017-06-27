data = load('data.mat');
lambda = linspace(400, 800, 401)';
omega = 2*pi./lambda;

values = [];
myspects = [];
%33.8_32.3_36.3_35.2_38.9
r1_1 = 47.5;
r2_1 = 45.3;
r3_1 = 60.6;
r4_1 = 61.8;
r5_1 = 37.5;
r6_1 = 49.6;
r7_1 = 47.8;
r8_1 = 55.9;
myname_1 = num2str(strcat('Desired :',num2str(r1_1),'-',num2str(r2_1),'-',num2str(r3_1),'-',num2str(r4_1),'-',num2str(r5_1),'-',num2str(r6_1),'-',num2str(r7_1),'-',num2str(r8_1)));
spect1 = scatter_0_generate_spectrum([r1_1,r2_1,r3_1,r4_1,r5_1,r6_1,r7_1,r8_1])

r1_2 = 49.35;
r2_2 = 45.0;
r3_2 = 58.9;
r4_2 = 61.8;
r5_2 = 38.1;
r6_2 = 49.6;
r7_2 = 47.6;
r8_2 = 55.6;

myname_2 = num2str(strcat('NN :',num2str(r1_2),'-',num2str(r2_2),'-',num2str(r3_2),'-',num2str(r4_2),'-',num2str(r5_2),'-',num2str(r6_2),'-',num2str(r7_2),'-',num2str(r8_2)));
spect2 = scatter_0_generate_spectrum([r1_2,r2_2,r3_2,r4_2,r5_2,r6_2,r7_2,r8_2])
r1_3 = 49.28;
r2_3 = 53.8;
r3_3 = 53.8;
r4_3 = 54.3;
r5_3 = 44.6;
r6_3 = 54.5;
r7_3 = 52.7;
r8_3 = 51.1;
myname_3 = num2str(strcat('MatLab :',num2str(r1_3),'-',num2str(r2_3),'-',num2str(r3_3),'-',num2str(r4_3),'-',num2str(r5_3),'-',num2str(r6_3),'-',num2str(r7_3),'-',num2str(r8_3)));
spect3 = scatter_0_generate_spectrum([r1_3,r2_3,r3_3,r4_3,r5_3,r6_3,r7_3,r8_3])
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

