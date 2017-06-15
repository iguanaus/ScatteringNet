data = load('data.mat');
lambda = linspace(400, 800, 401)';
omega = 2*pi./lambda;
eps_silver = interp1(data.omega_silver,data.epsilon_silver,omega);
eps_gold   = interp1(data.omega_gold,data.epsilon_gold,omega);

values = [];
myspects = [];
r1 = 30;
r2 = 30;
r3 = 70.0;
r4 = 38.6;
r5 = 54.5;
spect = scatter_0_generate_spectrum([r1,r2,r3,r4,r5]);
r1 = 80;
r2 = 25.6;
r3 = 80;
r4 = 35.6;
r5 = 50;
spect2 = scatter_0_generate_spectrum([r1,r2,r3,r4,r5]);
% r1 = 39.8;
% r2 = 17.6;
% r3 = 15.8;
% r4 = 47.7;
% r5 = 22.6;
% spect3 = run_spectrum_dielectric_advanced_seven(r1,r2,r3,r4,r5);
myspects = [myspects spect(1:2:401,1)];
values = [values ; [r1,r2,r3,r4,r5]];
%plot(lambda(1:5:501),[mylist',spect(1:5:501,1),spect2(1:5:501,1),spect3(1:5:501,1)]);%spect(1:5:501,1));
hold on
area([498,502],[4.0,4.0],'EdgeColor','none')
%area([690,710],[3.5,3.5],'EdgeColor','none')
alpha(.2)
%plot(lambda(1:2:399),[mylist',spect(1:2:399,1),spect2(1:2:399,1)])
plot(lambda(1:2:399),[spect(1:2:399,1),spect2(1:2:399,1)])

hold off

xlabel('Wavelength (nm)');
%ylabel('Cross Scattering Amplitude (normalized by power in dipole channel)');
ylabel('\sigma/\pi r^2');
%title('Residuals');
title('Geometries to match desired spectrums');
%legend('14/25/24/38 after 2 hours','14/25/24/38 after 6 hours');
%legend('Desired super-scattering at 465nm','Iteration One, 3053.4,30,30,30,30','Iteration Two, 33,35,28,25,26','Iteration Three, 19,60,33,51,10');
legend('Desired scattering',strcat('NN - Nanoparticle',num2str(r1),'/',num2str(r2),'/',num2str(r3),'/',num2str(r4)),'Matlab');%,'a');
csvwrite('test_dielectric.csv',myspects);
csvwrite('test_dielectric_val.csv',values);