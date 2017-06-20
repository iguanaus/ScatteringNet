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
x0 = [50,50,50,50,50]
A = [];
b = [];
Aeq = [];
beq = [];
lb = 30 * ones(1,5);
ub = 70 * ones(1,5);
nonlcon = [];

filename = 'spectrums/test_tio2_fixed33.8_32.3_36.3_35.2_38.9.csv';
myspect = csvread(filename);
save('var.mat','myspect')

options = optimoptions('fmincon','Display','iter','Algorithm','interior-point','FunctionTolerance',.05);
tic
%[x,fval] = patternsearch(@cost_function_try,x0,A,b,Aeq,beq,lb,ub,nonlcon,options)
fval = 200.0

%while fval > 100.0
r1 = round(rand*40+30,1);
r2 = round(rand*40+30,1);
r3 = round(rand*40+30,1);
r4 = round(rand*40+30,1);
r5 = round(rand*40+30,1);


x0 = [40,40,40,40,40]%r1,r2,r3,r4,r5]

filename = 'var.mat';
tmp = load(filename);
myspect = tmp.myspect;
cost_func = @(x)cost_function_mat(x,myspect)

[x,fval,exitflag,output] = fmincon(cost_func,x0,A,b,Aeq,beq,lb,ub,nonlcon, options)
%end
toc
