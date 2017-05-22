data = load('data.mat');
lambda = linspace(300, 800, 501)';
omega = 2*pi./lambda;
eps_silver = interp1(data.omega_silver,data.epsilon_silver,omega);
eps_gold   = interp1(data.omega_gold,data.epsilon_gold,omega);

values = [];
myspects = [];
r1 = 49;
r2 = 23;
r3 = 22;
r4 = 12;
r5 = 21;
spect1 = (2*pi)*run_spectrum_dielectric_advanced_six(r1,r2,r3,r4,r5)./(3*lambda.*lambda)
r1 = 20;
r2 = 20;
r3 = 20;
r4 = 20;
r5 = 20;
spect2 = (2*pi)*run_spectrum_dielectric_advanced_six(r1,r2,r3,r4,r5)./(3*lambda.*lambda)
r1 = 47.9;
r2 = 21.3;
r3 = 17.7;
r4 = 10.0;
r5 = 27.0;
spect3 = (2*pi)*run_spectrum_dielectric_advanced_six(r1,r2,r3,r4,r5)./(3*lambda.*lambda)
hold on
%plot(lambda,[spect1(1:1:end,1)-spect2(1:1:end,1),spect1(1:1:end,1)-spect3(1:1:end,1)]);%spect(1:5:501,1));
plot(lambda,[spect1(1:1:end,1),spect2(1:1:end,1),spect3(1:1:end,1)]);%spect(1:5:501,1));
%plot(lambda,spect2(1:1:end,1));
xlabel('Wavelength (nm)');
ylabel('Cross Scattering Amplitude (normalized by power in dipole channel)');
%title('Residuals');
title('Plotting what the NN returns for the closest geometry');
%legend('14/25/24/38 after 2 hours','14/25/24/38 after 6 hours');
legend('Desired for 49/23/22/12/21','NN begin training','NN for reveresed 48/21/18/10/27');

