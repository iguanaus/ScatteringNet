%This program will iterate through a sequence of starting points - randomly picked - then generate a file like the one I have for the excel doc.
lambda = linspace(400, 800, 401)';
omega = 2*pi./lambda;

eps_silica = 2.04*ones(length(omega), 1);
my_lam = lambda./1000;
eps_tio2 = 5.913+(.2441)*1./(my_lam.*my_lam-.0803);
eps_water  = 1.77*ones(length(omega), 1);
eps = [eps_silica eps_tio2 eps_silica eps_tio2 eps_silica eps_water];

wgts = cell(0);
bias = cell(0);
for i=0:4
    wgts{i+1} = transpose(load('spectrums/5_Layer_TiO2_200_layer/w_'+string(i)+'.txt'));
    bias{i+1} = load('spectrums/5_Layer_TiO2_200_layer/b_'+string(i)+'.txt');
end
dim = size(wgts);

options = optimoptions('fmincon','Display','iter','Algorithm','interior-point','ObjectiveLimit',.1,'SpecifyObjectiveGradient',true);


%cost_func_nn = @(x)cost_function_math_desired(x,wgts,bias,dim(2),50,70,omega,eps);
cost_func_nn = @(x)cost_function_nn_desired(x,wgts,bias,dim(2),50,100);

% start_params = [[50;50;50;50;50]]
% [mytime, convergence,x] = run_opt(start_params,cost_func_nn,options);
% mytime
% convergence
% x


%This is the actual computation
totconv = 0;
tottime = 0;
convergence_best = 1000.0;
yval = 0;
for i = 1:50
	start_params = all_start_params(:,i)

	[mytime, convergence,x] = run_opt(start_params,cost_func_nn,options);
	if convergence< convergence_best
		convergence_best = convergence
		myval = x;
	end
end
convergence_best
x


%3.255 best wit gradient


%3.1755 best without gradient