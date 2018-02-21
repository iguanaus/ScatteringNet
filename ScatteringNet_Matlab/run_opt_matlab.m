addpath(genpath('spherical_T_matrix'));
addpath(genpath('spectrums'));
addpath(genpath('dlib'));


%Compute the cost of matching the spectrum for 33,72,58,60,49
%x1 = [33,72,58,60,49];
%cost1 = cost_function_try(x1)

%This should be almost zero
%x2 = [33.6,72.3,58.4,63.4,49.2];
%cost2 = cost_function_try(x2)

% fun = @(x)min(x(2).^2-x(1).^2);
%x0 = 50 * ones(1,5);

lambda = linspace(400, 800, 401)';
omega = 2*pi./lambda;

eps_silica = 2.04*ones(length(omega), 1);
my_lam = lambda./1000;
eps_tio2 = 5.913+(.2441)*1./(my_lam.*my_lam-.0803);
eps_water  = 1.77*ones(length(omega), 1);
eps = [eps_silica eps_tio2 eps_silica eps_water];

x0 = [50,50,50]
A = [];
b = [];
Aeq = [];
beq = [];
lb = 30 * ones(1,5);
ub = 70 * ones(1,5);
nonlcon = [];

options = optimoptions('fmincon','Display','iter','Algorithm','interior-point','ObjectiveLimit',.1,'SpecifyObjectiveGradient',false);
tic
%[x,fval] = patternsearch(@cost_function_try,x0,A,b,Aeq,beq,lb,ub,nonlcon,options)
fval = 200.0

wgts = cell(0);
bias = cell(0);
for i=0:4
    wgts{i+1} = transpose(load('spectrums/3_Layer_TiO2_Final/w_'+string(i)+'.txt'));
    bias{i+1} = load('spectrums/3_Layer_TiO2_Final/b_'+string(i)+'.txt');
end
filename = 'spectrums/test_tio2_fixed_3/31.2_37.5_67.7.csv';
myspect = csvread(filename);
myspect = myspect(1:1:201,1);
dim = size(wgts);

%while fval > 100.0
r1 = round(rand*40+30,1);
r2 = round(rand*40+30,1);
r3 = round(rand*40+30,1);
r4 = round(rand*40+30,1);
r5 = round(rand*40+30,1);
r6 = round(rand*40+30,1);
r7 = round(rand*40+30,1);
r8 = round(rand*40+30,1);


%x0 = [r1,r2,r3,r4,r5,r6,r7,r8]
%x0 = [50;50;50;50;50;50;50;50];
%x0 = [49.5;47.4;47.9;42.3;50.3;50.4;62.7;61.8]
%x0 = [50,50,50,50,50,50,50,50]
x0 = [r1;r2;r3]
cost_func = @(x)cost_function_math(x,wgts,bias,dim(2),myspect,omega,eps)

[x,fval,exitflag,output] = fmincon(cost_func,x0,A,b,Aeq,beq,lb,ub,nonlcon, options)
%end
x
fval
toc
