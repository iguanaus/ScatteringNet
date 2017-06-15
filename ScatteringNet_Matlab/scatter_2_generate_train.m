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

%listOfRunnings1=[10 15 20 25 30 35]
%listOfRunnings2=[40 45 50 55 60]
%listOfRunnings =[10 15 20 25 30 35 40 45 50 55 60]
%listOfRunnings=[10 20]
low_bound = 30;
up_bound = 70;
num_iteration = 30000;
n = 0;
tic
while n < num_iteration
  n = n + 1;
  r1 = round(rand*40+30,1);
  r2 = round(rand*40+30,1);
  r3 = round(rand*40+30,1);
  r4 = round(rand*40+30,1);
  r5 = round(rand*40+30,1);
  spect = scatter_0_generate_spectrum([r1,r2,r3,r4,r5]);
  myspects = [myspects spect(1:2:401,1)];
  values = [values ; [r1,r2,r3,r4,r5]];
  if rem(n, 100) ==0;
    disp('On: ')
    disp(n)
    disp(num_iteration)
  end
end
toc
csvwrite('data/5_layer_tio2_fixed_2.csv',myspects);
csvwrite('data/5_layer_tio2_fixed_2_val.csv',values);