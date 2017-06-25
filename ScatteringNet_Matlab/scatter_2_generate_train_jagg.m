values = [];
myspects = [];

low_bound = 30;
up_bound = 70;
num_iteration = 40000;
n = 0;

num_layers = 3;

tic
while n < num_iteration
  n = n + 1;
  r = [];
  val = round(rand*(650-450)+450,0);
  for i = 1:num_layers
    r1 = round(rand*(up_bound-low_bound)+low_bound,1);
    r = [r r1];
  end
  spect = scatter_0_generate_spectrum_jagg(r,val);%,r6,r7,r8]);
  myspects = [myspects spect(1:2:401,1)];
  values = [values ; r];%,r6,r7,r8]];
  if rem(n, 100) ==0;
    disp('On: ')
    disp(n)
    disp(num_iteration)
  end
end
toc

csvwrite(strcat('data/jagg_layer_tio2_fixed_06_21_1.csv'),myspects);
csvwrite(strcat('data/jagg_layer_tio2_fixed_06_21_1_val.csv'),values);