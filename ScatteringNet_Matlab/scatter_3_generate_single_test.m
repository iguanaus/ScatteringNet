lambda = linspace(400, 800, 401)';

omega = 2*pi./lambda;
low_bound = 30;
up_bound = 70;

r1 = round(rand*(up_bound-low_bound)+low_bound,1);
r2 = round(rand*(up_bound-low_bound)+low_bound,1);
r3 = round(rand*(up_bound-low_bound)+low_bound,1);
r4 = round(rand*(up_bound-low_bound)+low_bound,1);
r5 = round(rand*(up_bound-low_bound)+low_bound,1);

spect = scatter_0_generate_spectrum([r1,r2,r3,r4,r5]);
myspects = [spect(1:2:401,1)];
values = [values ; [r1,r2,r3,r4,r5]];
plot(lambda(1:1:401),myspects);
xlabel('Wavelength (nm)');
ylabel('\sigma/\pi r^2');
title('Spectrum Example');
legend(values);
%values = [values ; [r1,r2,r3,r4,r5]];
%myname = num2str(strcat('test_dielectric_large_',num2str(r1),'_',num2str(r2),'_',num2str(r3),'_',num2str(r4),'_',num2str(r5)));
myname = num2str(strcat('spectrums/test_tio2_fixed',num2str(r1),'_',num2str(r2),'_',num2str(r3),'_',num2str(r4),'_',num2str(r5)));
csvwrite(strcat(myname,'.csv'),myspects);
csvwrite(strcat(myname,'_val.csv'),values);