% Path to the Matlab functions
addpath 'spherical_T_matrix';
addpath 'spherical_T_matrix/bessel';

% Wavelength of interest: 300 nm to 800 nm
lambda = linspace(400, 800, 401)';
omega = 2*pi./lambda;

% load data on epsilon
data = load('data.mat');
eps_silver = interp1(data.omega_silver,data.epsilon_silver,omega);
eps_gold   = interp1(data.omega_gold,data.epsilon_gold,omega);
eps_silica = 2.04*ones(length(omega), 1);
eps_water  = 1.77*ones(length(omega), 1);

eps_silica = 2.04*ones(length(omega), 1);
eps_tio2 = 8.04*ones(length(omega), 1);

% test case one: 40-nm-radius silver sphere in water
eps = [eps_silica eps_tio2 eps_silica eps_tio2 eps_silica eps_water];

a = [70,70,70,70,70];
cs_loworder = total_cs(a,omega,eps,7);
cs_highorder = total_cs(a,omega,eps,15);
dif = (cs_loworder(1:1:401,1)-cs_highorder(1:1:401,1))./cs_loworder(1:1:401,1);
plot(lambda, [dif]);
%spect = total_cs(a,omega,eps,3);
%spect2= total_cs(a,omega,eps,4);
%spect3= total_cs(a,omega,eps,5);
%spect4= total_cs(a,omega,eps,7);
%spect5= total_cs(a,omega,eps,10);
%plot(lambda, [spect(1:1:401,1),spect2(1:1:401,1),spect3(1:1:401,1),spect4(1:1:401,1),spect5(1:1:401,1)]);
legend('3','4','5','7','10');
xlabel('Wavelength (nm)');
%ylabel('\sigma/\pi r^2');
ylabel('\sigma');
title('Scattering of 240nm radi Versus Increasing Angular Order');
