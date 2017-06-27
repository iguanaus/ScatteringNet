%THis runs the specrum with a silver core fixed.
%spect(1:5:501,1)./(3*lambda.lambda)*2*pi
function spectrum = scatter_0_gen_spectrum_faster(r,omega,eps)

spectrum = total_cs(r,omega,eps,15)/(pi*sum(r)^2);
%spectrum = total_cs(r,omega,eps,9);