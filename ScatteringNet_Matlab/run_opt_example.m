function [time,convergence,x] = run_opt(start_params,cost_func,options)

A = [];
b = [];
Aeq = [];
beq = [];
lb = 30 * ones(1,5);
ub = 70 * ones(1,5);
nonlcon=[];


x0 = start_params;

tic;
[x,fval,exitflag,output] = fmincon(cost_func,x0,A,b,Aeq,beq,lb,ub,nonlcon, options);
x
time = toc;
convergence = fval;
