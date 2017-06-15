%Cost function for matching a desired spectrum to be used with NLOPT. 
%To make it faster import the spectrum outside the code (move the
%filename/myspect outside of the code).
%NOTE: TO RUN THIS YOU MUST IMPORT THE spherical_code and all subfolders.
function cost = cost_function_try(r)

%Change this to the spectrum you want it to match
% filename = 'spectrums/test_dielectric_large_67_67_33_52_41.csv';
% myspect = csvread(filename);
% save('var.mat','myspect')

filename = 'var.mat';
tmp = load(filename);
myspect = tmp.myspect;

spectrum_run = scatter_0_generate_spectrum(r);
spectrum_new = spectrum_run(1:2:401,1);
cost = sum((spectrum_new-myspect).^2);
%cost = sum(spectrum_new)./(spectrum_new(49)+spectrum_new(50)+spectrum_new(51));
end
