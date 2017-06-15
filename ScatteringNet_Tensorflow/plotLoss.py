#THis plots the values
import sys
import matplotlib.pyplot as plt
import numpy as np
#'results/Dielectric_Massive/train_train_loss_0.txt',
loss_files=['results/Dielectric_Order_BiasTest/train_val_loss_1_val_bias_test.txt','results/Dielectric_Order_BiasTest/train_val_loss_val_bias_test_sigmoid_nobias.txt']

lossValues = np.genfromtxt(loss_files[0],delimiter=',')
lossValues2 = np.genfromtxt(loss_files[1],delimiter=',')

#plt.plot(lossValues)
#plt.plot(lossValues2)



# #lossValues_oldbatch = np.genfromtxt('cum_loss_small10000.txt', delimiter=',')

# #print("Loss values:")
# #print(lossValues_newbatch)
# #print(len(lossValues_newbatch))
newVals = range(0,len(lossValues))
# #newVals1 = range(0,len(lossValues_newbatch_2))
newVals2 = range(0,len(lossValues2))
# newVals3 = range(0,len(lossValues_newbatch_4))
# newVals4 = range(0,len(lossValues_newbatch_5))
# newVals5 = range(0,len(lossValues_newbatch_6))

# #print len(newVals)
# #print len(newVals1)
# #print len(newVals2)
# #print len(newVals3)


#for i in range(0,len(newVals)):
# 	print("I val: " , i)
# 	newVals2[i] = i+len(newVals)

# for i in range(0,len(newVals5)):
# 	print("I val: " , i)
# 	newVals5[i] = i+len(newVals3)+len(newVals4)

# #plt.plot(lossValues_newbatch)
plt.plot(newVals,lossValues)
plt.plot(newVals2,lossValues2)
# plt.plot(newVals3,lossValues_newbatch_4)
# plt.plot(newVals4,lossValues_newbatch_5)
# plt.plot(newVals5,lossValues_newbatch_6)
plt.xlabel("Epochs trained (in 10's)")
plt.ylabel("MSE Validation Error")
plt.title("Validation error (25k params)")
plt.legend(['ReLu - Bias', 'Sigmoid'])
#plt.plot(lossValues_oldbatch)
plt.show()