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
    wgts{i+1} = transpose(load('spectrums/Dielectric_TiO2_5_06_20_2_new/w_'+string(i)+'.txt'));
    bias{i+1} = load('spectrums/Dielectric_TiO2_5_06_20_2_new/b_'+string(i)+'.txt');
end
filename = 'spectrums/TestTiO2Fixed/test_tio2_fixed67.9_36.7_52.1_40.7_43.1.csv';
myspect = csvread(filename);
myspect = myspect(1:1:201,1);
dim = size(wgts);

options = optimoptions('fmincon','Display','iter','Algorithm','interior-point','ObjectiveLimit',.1,'SpecifyObjectiveGradient',false);

%cost_func_nn = @(x)cost_function_math(x,wgts,bias,dim(2),myspect,omega,eps);
cost_func_nn = @(x)cost_function_nn(x,wgts,bias,dim(2),myspect);


%This is the actual computation
totconv = 0;
tottime = 0;
convergence_best = 1000.0;
myval = 0;
for i = 1:50
	start_params = all_start_params(:,i)

	[mytime, convergence,x] = run_opt(start_params,cost_func_nn,options);
	if convergence< convergence_best
		convergence_best = convergence
		myval = x;
	end


end



totconv/50
tottime/50