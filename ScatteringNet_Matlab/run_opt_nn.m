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



%options = optimoptions('fmincon','Display','iter','Algorithm','interior-point','FunctionTolerance',.05);
options = optimoptions('fmincon','Display','iter','Algorithm','interior-point','SpecifyObjectiveGradient',true);
tic
%[x,fval] = patternsearch(@cost_function_try,x0,A,b,Aeq,beq,lb,ub,nonlcon,options)
fval = 200.0

wgts = cell(0);
bias = cell(0);
for i=0:4
    wgts{i+1} = transpose(load('spectrums/3_Layer_TiO2_Final/w_'+string(i)+'.txt'));
    bias{i+1} = load('spectrums/3_Layer_TiO2_Final/b_'+string(i)+'.txt');
end
filename = 'spectrums/test_tio2_fixed_3/44.1_63.2_53.4.csv';
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
r9 = round(rand*40+30,1);
r10 = round(rand*40+30,1);



%x0 = [50;50;50;50;50;50;50;50];%r1,r2,r3,r4,r5]
x0 = [r1;r2;r3]
f_costa = @(x)cost_function_nn(x,wgts,bias,dim(2),myspect);
[x,fval,exitflag,output] = fmincon(f_costa,x0,A,b,Aeq,beq,lb,ub,nonlcon, options);
x
%[x,fval,exitflag,output] = fmincon(@f_costa_new,x0,A,b,Aeq,beq,lb,ub,nonlcon, options);
%end
toc
