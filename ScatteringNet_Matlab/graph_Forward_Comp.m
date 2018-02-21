xVals = [2;3;4;5;6;7;8;10]
matSpeed = [2.8782;3.7362;13.2708;19.2484;30.0606;35.951;56.084;92.76]
matUnc = [0.129449604093639;0.205383056750064;0.997562379001935;2.74948727947594;0.959539889738826;2.34082506821847;4.07405571881387;2.39575666543996]
nnSpeed =  [0.0229286;0.0316;0.152;0.1768;0.2778;0.3096;0.3464;0.42875]
nnUnc = [0.00633480112552873;0.00723187389270582;0.122582625196232;0.0651129787369615;0.0968230344494532;0.0529556418146358;0.0299132077851908;0.120680103993989]

fitXVals = linspace(2,20,100)
matYVals = 1.45.*fitXVals.*fitXVals -5.606.*fitXVals+8.15768
nnYVals = 0.04897087.*fitXVals -0.09048404


fig = figure;
hold on
errorbar(xVals,matSpeed,matUnc,'.');
plot(fitXVals,matYVals)

errorbar(xVals,nnSpeed,nnUnc,'.');
plot(fitXVals,nnYVals)
ax = get(fig,'CurrentAxes');
set(ax,'YScale','log')
hold off
%plot(result2)                              Desired Spectrum 
legend('Simulation Speed','Simulation Quadratic Fit','Neural Network Speed','Neural Net Linear Fit');
xlabel('Complexity (Number of Layers)');
ylabel('Runtime (s)');
title('Forward Runtime Versus Complexity');