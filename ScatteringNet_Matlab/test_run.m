% Path to the Matlab functions
addpath 'spherical_T_matrix';
addpath 'spherical_T_matrix/bessel';

% Wavelength of interest: 300 nm to 800 nm
lambda = linspace(300, 800, 501)';
omega = 2*pi./lambda;

% load data on epsilon
data = load('data.mat');
eps_silver = interp1(data.omega_silver,data.epsilon_silver,omega);
eps_gold   = interp1(data.omega_gold,data.epsilon_gold,omega);
eps_silica = 2.04*ones(length(omega), 1);
eps_water  = 1.77*ones(length(omega), 1);

% test case one: 40-nm-radius silver sphere in water
eps = [eps_silver eps_water];
a = [40];
plot(lambda, total_cs(a,omega,eps)/(pi*sum(a)^2));
legend('scattering', 'absorption');
xlabel('Wavelength (nm)');
ylabel('\sigma/\pi r^2');

%input('next?')
clf

%test case two: 16-nm-radius silica core, 5-nm-thick silver shell, in water
eps = [eps_silver eps_gold eps_water];
a = [40 40 5];
plot(lambda, total_cs(a,omega,eps)/(pi*sum(a)^2));
legend('scattering', 'absorption');
xlabel('Wavelength (nm)');
ylabel('\sigma/\pi r^2');
