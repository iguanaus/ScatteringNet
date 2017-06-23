data = load('data.mat');
lambda = linspace(400, 800, 401)';

values = [];
myspects = [];

for v1=[51.8];
    for v2=[54.7];
        for v3=[44.9];
            for v4=[48.7];
                 for v5=[62.7];
                    spect = scatter_0_generate_spectrum([v1,v2,v3,v4,v5]);
                    %spect(1:1:501,1)./(3*lambda.*lambda)*2*pi
                    myspects = [myspects spect(1:2:401,1)];%./(3*lambda.*lambda)*2*pi];
                    myname = num2str(strcat(num2str(v1),'--',num2str(v2),'--',num2str(v3),'--',num2str(v4),'--',num2str(v5)));
                    values = [values , string(myname)];

                end
            end
        end
    end
end
plot(lambda(1:2:401),myspects);%spect(1:5:501,1));
xlabel('Wavelength (nm)');
ylabel('\sigma/\pi r^2');
title('Train Data Set');
legend(values)
%csvwrite('test_large_fixed_five_temp.csv',myspects);
%csvwrite('test_large_fixed_five_val_temp.csv',values);
