addpath(genpath('spherical_T_matrix'));
addpath(genpath('spectrums'));
addpath(genpath('dlib'));


%Compute the cost of matching the spectrum for 33,72,58,60,49
x1 = [33,72,58,60,49];
cost1 = cost_function_try(x1)

%This should be almost zero
x2 = [33.6,72.3,58.4,63.4,49.2];
cost2 = cost_function_try(x2)

% fun = @(x)min(x(2).^2-x(1).^2);
%x0 = 50 * ones(1,5);
x0 = [50,50,50,50,50]
A = [];
b = [];
Aeq = [];
beq = [];
lb = 20 * ones(1,5);
ub = 80 * ones(1,5);
nonlcon = [];

filename = 'spectrums/test_dielectric_large_62.7_31.2_39.3_64.5_43.9.csv';
myspect = csvread(filename);
save('var.mat','myspect')

options = optimoptions('fmincon','Display','iter','Algorithm','active-set','OutputFcn',outfun);
tic
%[x,fval] = patternsearch(@cost_function_try,x0,A,b,Aeq,beq,lb,ub,nonlcon,options)
[x,fval,exitflag,output] = fmincon(@cost_function_try,x0,A,b,Aeq,beq,lb,ub,nonlcon, options)
toc

