data = load('data.mat');
lambda = linspace(400, 800, 401)';
omega = 2*pi./lambda;
eps_silver = interp1(data.omega_silver,data.epsilon_silver,omega);
eps_gold   = interp1(data.omega_gold,data.epsilon_gold,omega);
r1 = 100
r2 = 10
r3 = 100
r4 = 10
r5 = 200
r6 = 10
values = [];
myspects = [];
posVals = [10];
mymaxes = [];
for v1=[30 70];
    for v2=[30 70];
        for v3=[30];
            for v4=[30 70];
                 for v5=[30 70];
                    spect = real(scatter_0_generate_spectrum([v1,v2,v3,v4,v5]));
                    %spect(1:1:501,1)./(3*lambda.*lambda)*2*pi
                    myspects = [myspects spect(1:1:401,1)];%./(3*lambda.*lambda)*2*pi];
                    myname = num2str(strcat(num2str(v1),'--',num2str(v2),'--',num2str(v3),'--',num2str(v4),'--',num2str(v5)));
                    values = [values , string(myname)]; %[v1,v2,v3,v4]];
                    %end
                    %mymaxes = [mymaxes , max(spect)];
                end
            end
        end
    end
end
%values = [values ; [r1,r2,r3,r4,r5,r6]];

%spect(1:1:501,1)./(3*lambda.*lambda)*2*pi
plot(lambda(1:1:401),myspects);%spect(1:5:501,1));
xlabel('Wavelength (nm)');
ylabel('\sigma/\pi r^2');
title('Train Data Set - Fixed TiO2 Lossy');
legend(values)
%csvwrite('test_large_fixed_five_temp.csv',myspects);
%csvwrite('test_large_fixed_five_val_temp.csv',values);
%mymaxes