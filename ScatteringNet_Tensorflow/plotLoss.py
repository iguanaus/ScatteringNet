#THis plots the values
import sys
import matplotlib.pyplot as plt
import numpy as np
#'results/Dielectric_Massive/train_train_loss_0.txt',
loss_files=['results/Dielectric_Corrected_TiO2/train_val_loss_0_val.txt','results/Dielectric_Corrected_TiO2/train_val_loss_1_val.txt','results/Dielectric_Corrected_TiO2/train_val_loss_2_val.txt','results/Dielectric_Corrected_TiO2/train_val_loss_3_val.txt','results/Dielectric_Corrected_TiO2/train_val_loss_4_val.txt','results/Dielectric_Corrected_TiO2/train_val_loss_5_val.txt','results/Dielectric_Corrected_TiO2/train_val_loss_6_val.txt']

#Dielectric_Order_BiasTest/train_val_loss_1_val_bias_test.txt','results/Dielectric_Order_BiasTest/train_val_loss_val_bias_test_sigmoid_nobias.txt']

lossValues = np.genfromtxt(loss_files[0],delimiter=',')
lossValues2 = np.genfromtxt(loss_files[1],delimiter=',')
lossValues3 = np.genfromtxt(loss_files[2],delimiter=',')
lossValues4 = np.genfromtxt(loss_files[3],delimiter=',')
lossValues5 = np.genfromtxt(loss_files[4],delimiter=',')
lossValues6 = np.genfromtxt(loss_files[5],delimiter=',')

#plt.plot(lossValues)
#plt.plot(lossValues2)



# #lossValues_oldbatch = np.genfromtxt('cum_loss_small10000.txt', delimiter=',')

# #print("Loss values:")
# #print(lossValues_newbatch)
# #print(len(lossValues_newbatch))
newVals = range(0,len(lossValues))
# #newVals1 = range(0,len(lossValues_newbatch_2))
newVals2 = range(0,len(lossValues2))
newVals3 = range(0,len(lossValues3))
newVals4 = range(0,len(lossValues4))
newVals5 = range(0,len(lossValues5))
newVals6 = range(0,len(lossValues6))

# #print len(newVals)
# #print len(newVals1)
# #print len(newVals2)
# #print len(newVals3)


for i in range(0,len(newVals2)):
 	print("I val: " , i)
 	newVals2[i] = i+len(newVals)

for i in range(0,len(newVals3)):
 	print("I val: " , i)
 	newVals3[i] = i+len(newVals2)+len(newVals)
for i in range(0,len(newVals4)):
 	print("I val: " , i)
 	newVals4[i] = i+len(newVals2)+len(newVals)*2.0
for i in range(0,len(newVals5)):
 	print("I val: " , i)
 	newVals5[i] = i+len(newVals2)+len(newVals)*3.0
for i in range(0,len(newVals6)):
 	print("I val: " , i)
 	newVals6[i] = i+len(newVals2)+len(newVals)*4.0

# for i in range(0,len(newVals5)):
# 	print("I val: " , i)
# 	newVals5[i] = i+len(newVals3)+len(newVals4)

# #plt.plot(lossValues_newbatch)
plt.plot(newVals,lossValues)
plt.plot(newVals2,lossValues2)
plt.plot(newVals3,lossValues3)
plt.plot(newVals4,lossValues4)
plt.plot(newVals5,lossValues5)
r