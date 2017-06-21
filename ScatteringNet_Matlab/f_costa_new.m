function [cost,gradient] = f_costa_new(x)

  wgts = cell(0);
  bias = cell(0);
  for i=0:4
      wgts{i+1} = transpose(load('spectrums/Dielectric_TiO2_5_06_20/w_'+string(i)+'.txt'));
      bias{i+1} = load('spectrums/Dielectric_TiO2_5_06_20/b_'+string(i)+'.txt');
  end
  filename = 'var.mat';
  tmp = load(filename);
  myspect = tmp.myspect(1:1:201);
  dim = size(wgts);
  weights = wgts;
  biases = bias;
  depth = dim;
  filename = 'spectrums/TestTiO2Fixed/test_tio2_fixed33.8_32.3_36.3_35.2_38.9.csv';
  spectCompare = csvread(filename);


	input = x;
	layer = max(0,weights{1}*input)+biases{1};
  	for j=2:depth-1;
    	layer = max(0,weights{j}*layer)+biases{j};
  	end
  	[layer, Jacobian] = NN(weights,biases,x);
    %layer = weights{depth}*layer+biases{depth};
    
    
    %cost = sum(layer)./(layer(49)+layer(50)+layer(51));
    %cost = mean(layer)/mean(layer(50:60,:));
    cost = sum((spectCompare-layer).^2);
    gradient = Jacobian2Gradient(Jacobian,layer,spectCompare);
end
