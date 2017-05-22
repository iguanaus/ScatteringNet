#THis plots the values
import sys
import matplotlib.pyplot as plt
import numpy as np



#lossValues_newbatch = np.genfromtxt('results/DielectricExpanded/dieletric_loss_power_one.txt', delimiter=',')
#lossValues_newbatch_2 = np.genfromtxt('results/DielectricExpanded/dieletric_loss_power_two.txt', delimiter=',')
#lossValues_newbatch_3 = np.genfromtxt('results/DielectricExpanded/dieletric_loss_power_three.txt', delimiter=',')
lossValues_newbatch_4 = np.genfromtxt('results/DielectricExpanded/dieletric_loss_power_four_1.txt', delimiter=',')
lossValues_newbatch_5 = np.genfromtxt('results/DielectricExpanded/dieletric_loss_power_four.txt', delimiter=',')
lossValues_newbatch_6 = np.genfromtxt('results/DielectricExpanded/dieletric_loss_power_four_3.txt', delimiter=',')


#lossValues_oldbatch = np.genfromtxt('cum_loss_small10000.txt', delimiter=',')

#print("Loss values:")
#print(lossValues_newbatch)
#print(len(lossValues_newbatch))
#newVals = range(0,len(lossValues_newbatch))
#newVals1 = range(0,len(lossValues_newbatch_2))
#newVals2 = range(0,len(lossValues_newbatch_3))
newVals3 = range(0,len(lossValues_newbatch_4))
newVals4 = range(0,len(lossValues_newbatch_5))
newVals5 = range(0,len(lossValues_newbatch_6))

#print len(newVals)
#print len(newVals1)
#print len(newVals2)
#print len(newVals3)

for i in range(0,len(newVals4)):
	print("I val: " , i)
	newVals4[i] = i+len(newVals3)

for i in range(0,len(newVals5)):
	print("I val: " , i)
	newVals5[i] = i+len(newVals3)+len(newVals4)

#plt.plot(lossValues_newbatch)
#plt.plot(newVals1,lossValues_newbatch_2)
#plt.plot(newVals2,lossValues_newbatch_3)
plt.plot(newVals3,lossValues_newbatch_4)
plt.plot(newVals4,lossValues_newbatch_5)
plt.plot(newVals5,lossValues_newbatch_6)
plt.xlabel("Epochs trained (in 100's)")
plt.ylabel("MSE Error across one epoch and all data points on the output")
plt.title("Training error over time for ~1820 params")
#plt.plot(lossValues_oldbatch)
plt.show()