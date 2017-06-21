function [cost,gradient] = cost_function_nn(x,weights,biases,depth,spectCompare)
	input = x;
	layer = max(0,weights{1}*input)+biases{1};
  for j=2:depth-1;
    	layer = max(0,weights{j}*layer)+biases{j};
  end
  
  %[layer, Jacobian] = NN(weights,biases,x);
  layer = weights{depth}*layer+biases{depth};
  Jacobian = layer;
    
    
  %cost = sum(layer)./(layer(49)+layer(50)+layer(51));
  %cost = mean(layer)/mean(layer(50:60,:));
  cost = sum((spectCompare-layer).^2);
  gradient = Jacobian2Gradient(Jacobian,layer,spectCompare)*2.0;
end
