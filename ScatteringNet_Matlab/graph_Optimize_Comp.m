xVals = [2;4;6;8]
nnSpeed = [0.0392;0.316;0.824;1.13]
nnUnc = [0.00425;0.272;0.135;0.086]
matSpeed = [0.765;12.64;106.22;313.25]
matUnc = [0.042;1.91;7.8;43.5]

fitXVals = linspace(2,20,100)
matYVals = .0355.*fitXVals.^4.41
nnYVals = .1835.*fitXVals-.328
%1.45.*fitXVals.*fitXVals -5.606.*fitXVals+8.15768
%nnYVals = 0.04897087.*fitXVals -0.09048404


fig = figure;
hold on
errorbar(xVals,matSpeed,matUnc,'.');
plot(fitXVals,matYVals)

errorbar(xVals,nnSpeed,nnUnc,'.');
plot(fitXVals,nnYVals)
ax = get(fig,'CurrentAxes');
%set(ax,'YScale','log')
hold off
%plot(result2)                              Desired Spectrum 
legend('Simulation Speed','Simulation Power Fit','Neural Network Speed','Neural Net Linear Fit');
xlabel('Complexity (Number of Layers)');
ylabel('Runtime (s)');
title('Inverse Design Runtime Versus Complexity');