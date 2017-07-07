all_start_params = []
for i = 1:100
	r1 = round(rand*40+30,1);
	r2 = round(rand*40+30,1);
	r3 = round(rand*40+30,1);
	r4 = round(rand*40+30,1);
	r5 = round(rand*40+30,1);
	r6 = round(rand*40+30,1);
	r7 = round(rand*40+30,1);
	r8 = round(rand*40+30,1);
	
	start_params = [r1;r2;r3;r4;r5;r6]%;r7;r8];%;r4;r5;r6;r7];
	all_start_params = [all_start_params , start_params]
end
