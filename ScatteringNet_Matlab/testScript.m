x1 = linspace(0,1);
x2 = linspace(3/4,1);
y1 = sin(2*pi*x1);
y2 = sin(2*pi*x2);
figure(1)
% plot on large axes
plot(x1,y1)
% create smaller axes in top right, and plot on it
axes('Position',[.7 .7 .2 .2])
box on
plot(x2,y2)