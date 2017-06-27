%This program will iterate through a sequence of starting points - randomly picked - then generate a file like the one I have for the excel doc.
lambda = linspace(400, 800, 401)';
omega = 2*pi./lambda;

eps_silica = 2.04*ones(length(omega), 1);
my_lam = lambda./1000;
eps_tio2 = 5.913+(.2441)*1./(my_lam.*my_lam-.0803);
eps_water  = 1.77*ones(length(omega), 1);
eps = [eps_silica eps_tio2 eps_silica eps_tio2 eps_silica eps_tio2 eps_silica eps_tio2 eps_water];

wgts = cell(0);
bias = cell(0);
for i=0:4
    wgts{i+1} = transpose(load(strcat('spectrums/8_Layer_TiO2_Final_3/w_',num2str(i),'.txt')));
    bias{i+1} = load(strcat('spectrums/8_Layer_TiO2_Final_3/b_',num2str(i),'.txt'));
end
filename = 'spectrums/test_tio2_fixed_8/47.5_45.3_60.6_61.8_37.5_49.6_47.8_55.9.csv';
myspect = csvread(filename);
myspect = myspect(1:1:200,1);
dim = size(wgts);

options = optimoptions('fmincon','Display','iter','Algorithm','interior-point','ObjectiveLimit',1.0,'SpecifyObjectiveGradient',true);

%cost_func_nn = @(x)cost_function_math(x,wgts,bias,dim(2),myspect,omega,eps);
cost_func_nn = @(x)cost_function_nn(x,wgts,bias,dim(2),myspect);


%This is the actual computation
totconv = 0;
tottime = 0;

for i = 1:25
	start_params = all_start_params(:,i)
	%r1 = round(rand*40+30,1);
	%r2 = round(rand*40+30,1);
	%r3 = round(rand*40+30,1);
	%r4 = round(rand*40+30,1);
	%r5 = round(rand*40+30,1);
	%start_params = [r1;r2;r3;r4;r5];

	[mytime, convergence] = run_opt(start_params,cost_func_nn,options);
	mytime;
	if (convergence < 1.0)
		convergence = 1.0;
	else
		convergence = 0.0;
	end
	convergence;
	tottime = tottime + mytime;
	totconv = totconv + convergence;
	i
end
totconv/25
tottime/25