data = load('data.mat');
lambda = linspace(400, 800, 401)';
omega = 2*pi./lambda;

values = [];
myspects = [];
r1 = 70.0;
r2 = 30.2;
r3 = 44.3;
r4 = 60.0;
r5 = 44.1;
spect = scatter_0_generate_spectrum([r1,r2,r3,r4,r5]);
r1 = 70.0;
r2 = 52.8;
r3 = 70.0;
r4 = 34.1;
r5 = 30.0;
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
area([500,520],[4.0,4.0],'EdgeColor','none')
%area([690,710],[3.5,3.5],'EdgeColor','none')
alpha(.2)
%plot(lambda(1:2:399),[mylist',spect(1:2:399,1),spect2(1:2:399,1)])
plot(lambda(1:2:399),[spect(1:2:399,1),spect2(1:2:399,1)])

hold off

xlabel('Wavelength (nm)');
ylabel('\sigma/\pi r^2');
%title('Residuals');
title('Geometries to match desired spectrums');
legend('Desired scattering',strcat('NN - Nanoparticle',num2str(r1),'/',num2str(r2),'/',num2str(r3),'/',num2str(r4)),'Matlab');%,'a');
csvwrite('test_dielectric.csv',myspects);
csvwrite('test_dielectric_val.csv',values);