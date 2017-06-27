%Cost function for matching a desired spectrum to be used with NLOPT. 
%To make it faster import the spectrum outside the code (move the
%filename/myspect outside of the code).
%NOTE: TO RUN THIS YOU MUST IMPORT THE spherical_code and all subfolders.
function [cost,gradient] = cost_function_math(r,weights,biases,depth,spectToCompare,omega,eps)

%Change this to the spectrum you want it to match
% filename = 'spectrums/test_dielectric_large_67_67_33_52_41.csv';
% myspect = csvread(filename);
% save('var.mat','myspect')
[layer, Jacobian] = NN(weights,biases,r);
%length(layer)
spectrum_run = scatter_0_gen_spectrum_faster(r,omega,eps);
spectrum_new = spectrum_run(1:2:399,1);
%length(spectrum_new)
%length(spectToCompare)
cost = sum((spectrum_new-spectToCompare).^2);
gradient = Jacobian2Gradient(Jacobian,spectrum_new,spectToCompare)*2.0;
%gradient
%gradient = spectrum_new;
%cost = mean(spectrum_new)/mean(spectrum_new(50:60,:));
    
%cost = sum(spectrum_new)./(spectrum_new(49)+spectrum_new(50)+spectrum_new(51));
end
