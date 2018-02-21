function [cost,gradient] = cost_function_nn_desired(x,weights,biases,depth,desiredLowVal,desiredUpVal,omega,eps)
	%input = x;
	%layer = max(0,weights{1}*input)+biases{1};
  %for j=2:depth-1;
  %  	layer = max(0,weights{j}*layer)+biases{j};
  %end
  N=201;
  vec=zeros(N,1);
  positions=[desiredLowVal:desiredUpVal];
  vec(positions)=1; 
  %This is thus the multiplicative thing 

  [layer, Jacobian] = NN(weights,biases,x);
  spectrum_run = scatter_0_gen_spectrum_faster(x,omega,eps);
  spectrum_new = spectrum_run(1:2:401,1);

  %layer = weights{depth}*layer+biases{depth};
  %Jacobian = layer;
  %cost = sum(layer)./(layer(49)+layer(50)+layer(51));
  %cost = mean(layer)/mean(layer(50:60,:));
  topVal = sum(spectrum_new);
  botVal = sum(spectrum_new.*vec);
  cost = topVal/botVal;
                  %These live when it is a value
  scalingFactor = vec + abs(1-vec)./topVal;

  %cost = mean(layer)/mean(layer(desiredLowVal:desiredUpVal,:));
  %The gradient is slightly weird, but effectively it is 
  %Similar to before, I need a list of 0's and 1's, and we can go from there.
  %saclingFactor = 
  gradient = transpose(Jacobian)*scalingFactor ;
  %gradient = Jacobian2Gradient(Jacobian,layer,spectCompare)*2.0;
end
