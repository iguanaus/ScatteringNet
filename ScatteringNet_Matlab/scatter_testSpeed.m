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
num_iteration = 1000;
n = 0;
r1 = round(rand*40+30,1);
r2 = round(rand*40+30,1);
r3 = round(rand*40+30,1);
r4 = round(rand*40+30,1);
r5 = round(rand*40+30,1);
r6 = round(rand*40+30,1);
r7 = round(rand*40+30,1);
r8 = round(rand*40+30,1);

lambda = linspace(400, 800, 401)';
omega = 2*pi./lambda;
data = load('data.mat');
eps_silica = 2.04*ones(length(omega), 1);
my_lam = lambda./1000;
eps_tio2 = 5.913+(.2441)*1./(my_lam.*my_lam-.0803);
eps = [eps_silica eps_tio2 eps_silica eps_tio2 eps_silica eps_water];

wgts = cell(0);
bias = cell(0);
for i=0:4
    wgts{i+1} = transpose(load('spectrums/8_Layer_TiO2_Final_3/w_'+string(i)+'.txt'));
    bias{i+1} = load('spectrums/8_Layer_TiO2_Final_3/b_'+string(i)+'.txt');
end
weights = wgts;
input = [50;50;50;50;50;50;50;50];
biases = bias;
depth = size(wgts);
depth = depth(2);
chain = cell(0);
tic

while n < num_iteration
  [layer,grad] = NN(weights,biases,input);
  %layer = max(0,weights{1}*input)+biases{1};
  %for j=2:depth-1;
  %  layer = max(0,weights{i}*layer)+biases{i};
  %end
  %layer = weights{depth}*layer+biases{depth};

  n = n + 1;
  %spectrum = total_cs([r1,r2,r3,r4,r5,r6,r7,r8],omega,eps,15)/(pi*sum([r1,r2,r3,r4,r5,r6,r7,r8])^2);
  %r6 = round(rand*40+30,1);
  %r7 = round(rand*40+30,1);
  %r8 = round(rand*40+30,1);
  %spect = scatter_0_generate_spectrum([r1,r2,r3,r4,r5]);%,r6,r7,r8]);
  %myspects = [myspects spect(1:2:401,1)];
  %values = [values ; [r1,r2,r3,r4,r5]];%,r6,r7,r8]];
  if rem(n, 1000) ==0;
    disp('On: ')
    disp(n)
    disp(num_iteration)
  end
end
toc
csvwrite('data/8_layer_tio2_fixed_1_2.csv',myspects);

csvwrite('data/8_layer_tio2_fixed_1_val_2.csv',values);