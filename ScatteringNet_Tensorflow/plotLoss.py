#THis plots the values
import sys
import matplotlib.pyplot as plt
import numpy as np
#'results/Dielectric_Massive/train_train_loss_0.txt',
loss_files=['results/Dielectric_TiO2_6_22_dropout/train_train_loss_0_val.txt','results/Dielectric_TiO2_6_22_dropout/train_train_loss_1_val.txt','results/Dielectric_TiO2_6_22_dropout/train_train_loss_2_val.txt','results/Dielectric_TiO2_6_22_dropout/train_train_loss_3_val.txt']

loss_files_2=['results/Dielectric_TiO2_5_06_20_2_new_100/train_train_loss_0_val.txt','results/Dielectric_TiO2_5_06_20_2_new_100/train_train_loss_1_val.txt','results/Dielectric_TiO2_5_06_20_2_new_100/train_train_loss_2_val.txt','results/Dielectric_TiO2_5_06_20_2_new_100/train_train_loss_3_val.txt']

#Dielectric_Order_BiasTest/train_val_loss_1_val_bias_test.txt','results/Dielectric_Order_BiasTest/train_val_loss_val_bias_test_sigmoid_nobias.txt']

lossValues = np.genfromtxt(loss_files[0],delimiter=',')
lossValues2 = np.genfromtxt(loss_files[1],delimiter=',')
lossValues3 = np.genfromtxt(loss_files[2],delimiter=',')
lossValues4 = np.genfromtxt(loss_files[3],delimiter=',')
lossValues = np.append(lossValues,[lossValues2,lossValues3,lossValues4])
#print(lossValues+lossValues2)



lossValues_2 = np.genfromtxt(loss_files_2[0],delimiter=',')
lossValues_22 = np.genfromtxt(loss_files_2[1],delimiter=',')
lossValues_23 = np.genfromtxt(loss_files_2[2],delimiter=',')
lossValues_24 = np.genfromtxt(loss_files_2[3],delimiter=',')
lossValues_2 = np.append(lossValues_2,[lossValues_22,lossValues_23,lossValues_24])
#plt.plot(lossValues)
#plt.plot(lossValues2)

plt.plot(lossValues)

plt.plot(lossValues_2)

plt.show()