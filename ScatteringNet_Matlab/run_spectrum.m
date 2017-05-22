function spectrum = run_spectrum(thickness)
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
a = [thickness];
spectrum = total_cs(a,omega,eps)/(pi*sum(a)^2);
