values = [];
myspects = [];

low_bound = 30;
up_bound = 70;
num_iteration = 20000;
n = 0;

tic
while n < num_iteration
  n = n + 1;
  r1 = round(rand*(up_bound-low_bound)+low_bound,1);
  r2 = round(rand*(up_bound-low_bound)+low_bound,1);
  r3 = round(rand*(up_bound-low_bound)+low_bound,1);
  r4 = round(rand*(up_bound-low_bound)+low_bound,1);
  r5 = round(rand*(up_bound-low_bound)+low_bound,1);
  spect = scatter_0_generate_spectrum([r1,r2,r3,r4,r5]);%,r6,r7,r8]);
  myspects = [myspects spect(1:2:401,1)];
  values = [values ; [r1,r2,r3,r4,r5]];%,r6,r7,r8]];
  if rem(n, 1000) ==0;
    disp('On: ')
    disp(n)
    disp(num_iteration)
  end
end
toc
csvwrite('data/5_layer_tio2_fixed_06_20.csv',myspects);
csvwrite('data/5_layer_tio2_fixed_06_20_val.csv',values);