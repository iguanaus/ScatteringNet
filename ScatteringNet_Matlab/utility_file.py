import numpy as np

def fixEndings():
	f = open('order_die_1_val.csv','r')
	out = open('order_die_1_val_corrected.csv','w')

	i = 30

	for line in f:
		out.write(line[0:11]+"," + str(i) + "\n")
		i += 5
		if (i == 75):
			i = 30

	out.flush()

	print("Done")

def fixXValues():

	#f = open('order_die_1.csv','r')
	out = open('order_die_1_corrected.csv','w')

	train_Y = np.genfromtxt('order_die_1.csv',delimiter=',')

	#For each row divide by the wavelength. The wavelength is 400,402,404,......800
	step = -1
	for i in xrange(400,800,2):
		step += 1 

		newline = train_Y[step]/(2*3.14159)*(3.0*i*i)
		print(newline.shape)
		out.write(','.join(map(str,list(newline)))+"\n")


	#print(train_Y)
	#print(train_Y[0])
	#print(train_Y[0]/200.0)

#This combines it vertically - stacks them
def combineFile():
	file1 = open('data/10_layer_tio2_combined_06_21_val.csv','r')
	file2 = open('data/10_layer_tio2_combined_06_22_complete_val.csv','r')
	out = open('data/10_layer_tio2_combined_80000_val.csv','w')


	for line in file1:
		out.write(line)
	for line in file2:
		out.write(line)


	out.flush()

	print("Done")


#This combines them horizontally
def combineFileHorizontally():
	#f = open('order_die_1.csv','r')
	out = open('data/10_layer_tio2_combined_80000.csv','w')

	train_Y_1 = np.genfromtxt('data/10_layer_tio2_combined_06_21.csv',delimiter=',')
	train_Y_2 = np.genfromtxt('data/10_layer_tio2_combined_06_22_complete.csv',delimiter=',')

	#For each row divide by the wavelength. The wavelength is 400,402,404,......800
	step = -1
	for i in xrange(400,802,2):
		step += 1 

		newline = np.concatenate((train_Y_1[step],train_Y_2[step]),axis=0)
		print(newline.shape)
		out.write(','.join(map(str,list(newline)))+"\n")

combineFile()


combineFileHorizontally()


