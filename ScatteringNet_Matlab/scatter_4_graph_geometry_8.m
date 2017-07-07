data = load('data.mat');
lambda = linspace(400, 800, 401)';
omega = 2*pi./lambda;

values = [];
myspects = [];
%33.8_32.3_36.3_35.2_38.9
r1_1 = 58.4;
r2_1 = 60.2;
r3_1 = 41.0;
r4_1 = 57.2;
r5_1 = 56.2;
r6_1 = 36.5;
r7_1 = 34.8;
r8_1 = 49.9;
myname_1 = num2str(strcat('Desired :',num2str(r1_1),'-',num2str(r2_1),'-',num2str(r3_1),'-',num2str(r4_1),'-',num2str(r5_1),'-',num2str(r6_1),'-',num2str(r7_1),'-',num2str(r8_1)));
spect1 = scatter_0_generate_spectrum([r1_1,r2_1,r3_1,r4_1,r5_1,r6_1,r7_1,r8_1])

r1_2 = 61.0;
r2_2 = 66.2;
r3_2 = 39.5;
r4_2 = 51.3;
r5_2 = 49.3;
r6_2 = 34.9;
r7_2 = 37.8;
r8_2 = 52.6;

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
%plot(lambda,[spect1(1:1:end,1),spect2(1:1:end,1),spect3(1:1:end,1)]);%spect(1:5:501,1));
plot(lambda,spect1(1:1:end,1));
xlabel('Wavelength (nm)');
ylabel('\sigma/\pi r^2');
%title('Residuals');
title('Inverse Design');
%legend('14/25/24/38 after 2 hours','14/25/24/38 after 6 hours');
legend('Desired');
%legend(myname_1,myname_2,myname_3);
%legend('Desired for 49/23/22/12/21','NN begin training','NN for reveresed 48/21/18/10/27');

