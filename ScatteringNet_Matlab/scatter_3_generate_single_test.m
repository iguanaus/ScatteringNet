lambda = linspace(400, 800, 401)';

omega = 2*pi./lambda;
low_bound = 30;
up_bound = 70;

values = [];
r1 = round(rand*(up_bound-low_bound)+low_bound,1);
r2 = round(rand*(up_bound-low_bound)+low_bound,1);
r3 = round(rand*(up_bound-low_bound)+low_bound,1);
r4 = round(rand*(up_bound-low_bound)+low_bound,1);
r5 = round(rand*(up_bound-low_bound)+low_bound,1);
r6 = round(rand*(up_bound-low_bound)+low_bound,1);
r7 = round(rand*(up_bound-low_bound)+low_bound,1);
r8 = round(rand*(up_bound-low_bound)+low_bound,1);
r9 = round(rand*(up_bound-low_bound)+low_bound,1);
r10 = round(rand*(up_bound-low_bound)+low_bound,1);
r1 = 44.1; 
r2 = 63.2;
r3 = 53.4;
%r1 = 62.6
%r2 = 66.2
%r3 = 35.1
%r4 = 66.5
%r5 = 55.3
%r6 = 33.9
%r7 = 41.1
%r8 = 51.9

spect = scatter_0_generate_spectrum([r1,r2,r3]);
myspects = [spect(1:2:401,1)];
values = [values ; [r1,r2,r3]];
plot(lambda(1:2:401),myspects);
xlabel('Wavelength (nm)');
ylabel('\sigma/\pi r^2');
title('Spectrum Example');
%legend([[r1,r2,r3,r4,r5,r6,r7,r8]]);
%values = [values ; [r1,r2,r3,r4,r5]];
%myname = num2str(strcat('test_dielectric_large_',num2str(r1),'_',num2str(r2),'_',num2str(r3),'_',num2str(r4),'_',num2str(r5)));
myname = num2str(strcat('spectrums/test_tio2_fixed_3/',num2str(r1),'_',num2str(r2),'_',num2str(r3)));
csvwrite(strcat(myname,'.csv'),myspects);
csvwrite(strcat(myname,'_val.csv'),values);