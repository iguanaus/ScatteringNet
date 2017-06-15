values = [];
myspects = [];
c1 = 0;
c2 = 0;
data = load('data.mat');
lambda = linspace(400, 800, 401)';
omega = 2*pi./lambda;
eps_silver = interp1(data.omega_silver,data.epsilon_silver,omega);
eps_gold   = interp1(data.omega_gold,data.epsilon_gold,omega);

eps_silver = interp1(data.omega_silver,data.epsilon_silver,omega);
eps_silica = 2.04*ones(length(omega), 1);
eps_water  = 1.77*ones(length(omega), 1);
% load data on epsilon

% test case one: 40-nm-radius silver sphere in water
eps = [eps_silver eps_silica eps_silver eps_silica eps_silver eps_water];

listOfRunnings1=[10 15 20 25 30 35]
%listOfRunnings2=[40 45 50 55 60]
listOfRunnings =[10 15 20 25 30 35 40 45 50 55 60]
%listOfRunnings=[10 20]
r1=33;
r2=60;
r3=52;
r4=45;
r5=68;
spect = scatter_0_generate_spectrum([r1,r2,r3,r4,r5]);
myspects = [myspects spect(1:2:401,1)];
values = [values ; [r1,r2,r3,r4,r5]];
plot(myspects)
%values = [values ; [r1,r2,r3,r4,r5]];
%myname = num2str(strcat('test_dielectric_large_',num2str(r1),'_',num2str(r2),'_',num2str(r3),'_',num2str(r4),'_',num2str(r5)));
myname = num2str(strcat('spectrums/test_dielectric_large_',num2str(r1),'_',num2str(r2),'_',num2str(r3),'_',num2str(r4),'_',num2str(r5)));
csvwrite(strcat(myname,'.csv'),myspects);
csvwrite(strcat(myname,'_val.csv'),values);