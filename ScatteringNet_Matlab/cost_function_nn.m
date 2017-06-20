function cost = cost_function_nn(x,weights,biases,depth,spectCompare)
	input = x;
	layer = max(0,weights{1}*input)+biases{1};
  	for j=2:depth-1;
    	layer = max(0,weights{j}*layer)+biases{j};
  	end
  	layer = weights{depth}*layer+biases{depth};
    cost = sum((spectCompare-layer).^2);
end
