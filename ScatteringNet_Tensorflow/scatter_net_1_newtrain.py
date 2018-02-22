'''
    This program trains a feed-forward neural network. It takes in a geometric design (the radi of concentric spheres), and outputs the scattering spectrum. It is meant to be the first program run, to first train the weights. 
'''

import tensorflow as tf
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import os
import time
import argparse, os
import sys
RANDOM_SEED = 42
tf.set_random_seed(RANDOM_SEED)

num_decay = 43200

def init_weights(shape):
    """ Weight initialization """
    weights = tf.random_normal(shape, stddev=.1)
    return tf.Variable(weights)

def init_bias(shape):
    """ Weight initialization """
    biases = tf.random_normal([shape], stddev=.1)
    return tf.Variable(biases)

def save_weights(weights,biases,output_folder,weight_name_save,num_layers):
    for i in xrange(0, num_layers+1):
        weight_i = weights[i].eval()
        np.savetxt(output_folder+weight_name_save+"w_"+str(i)+".txt",weight_i,delimiter=',')
        bias_i = biases[i].eval()
        np.savetxt(output_folder+weight_name_save+"b_"+str(i)+".txt",bias_i,delimiter=',')
    return

def load_weights(output_folder,weight_load_name,num_layers):
    weights = []
    biases = []
    for i in xrange(0, num_layers+1):
        weight_i = np.loadtxt(output_folder+weight_load_name+"w_"+str(i)+".txt",delimiter=',')
        w_i = tf.Variable(weight_i,dtype=tf.float32)
        weights.append(w_i)
        bias_i = np.loadtxt(output_folder+weight_load_name+"b_"+str(i)+".txt",delimiter=',')
        b_i = tf.Variable(bias_i,dtype=tf.float32)
        biases.append(b_i)
    return weights , biases

def forwardprop(X, weights, biases, num_layers,dropout=False):
    htemp = None
    for i in xrange(0, num_layers):
        if i ==0:
            htemp = tf.nn.relu(tf.add(tf.matmul(X,weights[i]),biases[i])) 
        else:   
            htemp = tf.nn.relu(tf.add(tf.matmul(htemp,weights[i]),biases[i]))
    yval = tf.add(tf.matmul(htemp,weights[-1]),biases[-1])
    return yval

#This method reads from the 'X' and 'Y' file and gives in the input as an array of arrays (aka if the input dim is 5 and there are 10 training sets, the input is a 10X 5 array)
#a a a a a       3 3 3 3 3 
#b b b b b       4 4 4 4 4
#c c c c c       5 5 5 5 5

def get_data(data,percentTest=.2,random_state=42):
    x_file = data+"_val.csv"
    y_file = data+".csv"
    print("Train X: " , np.genfromtxt(x_file,delimiter=','))
    print(np.genfromtxt(x_file,delimiter=',').shape)
    print(np.genfromtxt(y_file,delimiter=',').shape)

    
    train_X = np.genfromtxt(x_file,delimiter=',')#[0:20000,:]
    print(train_X)
    train_Y = np.transpose(np.genfromtxt(y_file,delimiter=','))#[0:20000,:]
    print(train_Y)

    #Normalize train_X and train_Y.
    std_X = (train_X-train_X.mean(axis=0))/train_X.std(axis=0)
    std_Y = (train_Y-train_Y.mean(axis=0))/train_Y.std(axis=0)
    std_X = train_X
    std_Y = train_Y
    #for ele in train_Y:
    #    print len(ele)
    #    print ele
    X_train, X_test, y_train, y_test = train_test_split(std_X,std_Y,test_size=percentTest,random_state=random_state)
    #Now chunk the val in half 
    X_test, X_val, y_test, y_val = train_test_split(X_test,y_test,test_size=0.5,random_state=random_state)
    print("Train:")
    print(X_train)
    print(y_train)
    print("Test:")
    print(X_test)
    print(y_test)
    print("Val:")
    print(X_val)
    print(y_val)

    return X_train, y_train, X_test, y_test, X_val, y_val

def main(data,reuse_weights,output_folder,weight_name_save,weight_name_load,n_batch,numEpochs,lr_rate,lr_decay,num_layers,n_hidden,percent_val):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    train_X, train_Y , test_X, test_Y, val_X, val_Y = get_data(data,percentTest=percent_val)

    x_size = train_X.shape[1]
    y_size = train_Y.shape[1]

    # Symbols
    X = tf.placeholder("float", shape=[None, x_size])
    y = tf.placeholder("float", shape=[None, y_size])
    weights = []
    biases = []
    # Weight initializations
    if reuse_weights:
        (weights, biases) = load_weights(output_folder,weight_name_load,num_layers)

    else:
        for i in xrange(0,num_layers):
            if i ==0:
                weights.append(init_weights((x_size,n_hidden)))
            else:
                weights.append(init_weights((n_hidden,n_hidden)))
            biases.append(init_bias(n_hidden))
        weights.append(init_weights((n_hidden,y_size)))
        biases.append(init_bias(y_size))
    # Forward propagation
    yhat    = forwardprop(X, weights,biases,num_layers)
    
    # Backward propagation
    cost = tf.reduce_sum(tf.square(y-yhat))
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(lr_rate,global_step,int(train_X.shape[0]/n_batch),lr_decay,staircase=False)

    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost,global_step=global_step)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        step = 0
        curEpoch=0
        cum_loss = 0
        numFile = 0 
        while True:
            train_file_name = output_folder+"train_train_loss_" + str(numFile) + ".txt"
            if os.path.isfile(train_file_name):
                numFile += 1
            else:
                break
        train_loss_file = open(train_file_name,'w')
        test_loss_file = open(output_folder+"_train_loss_"+str(numFile) + "_val.txt",'w')
        val_loss_file = open(output_folder+"_val_loss_" + str(numFile) + "_val.txt",'w')
        start_time=time.time()
        print("========                         Iterations started                  ========")
        while curEpoch < numEpochs:
            batch_x = train_X[step * n_batch : (step+1) * n_batch]
            batch_y = train_Y[step * n_batch : (step+1) * n_batch]
            sess.run(optimizer, feed_dict={X: batch_x, y: batch_y})
            cum_loss += sess.run(cost,feed_dict={X:batch_x,y:batch_y})
            step += 1
            if step > int(train_X.shape[0]/n_batch): #Epoch finished
                step = 0
                curEpoch +=1            
                train_loss_file.write(str(float(cum_loss))+str("\n"))
                if (curEpoch % 10 == 0 or curEpoch == 1):
                    #Calculate the validation loss
                    test_loss = sess.run(cost,feed_dict={X:test_X,y:test_Y})
                    print("Test loss: " , str(test_loss))
                    test_loss_file.write(str(float(test_loss))+str("\n"))
                    test_loss_file.flush()
                    print("Epoch: " + str(curEpoch+1) + " : Loss: " + str(cum_loss))
                    train_loss_file.flush()
                cum_loss = 0
        #Now do validation
        val_loss = sess.run(cost,feed_dict={X:val_X,y:val_Y})
        print("Validation loss: " , str(val_loss))
        val_loss_file.write(str(float(val_loss))+str("\n"))
        val_loss_file.flush()

        save_weights(weights,biases,output_folder,weight_name_save,num_layers)

    print "========Iterations completed in : " + str(time.time()-start_time) + " ========"
    sess.close()

if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description="Physics Net Training")
    #parser.add_argument("--data",type=str,default='/Users/johnpeurifoy/Documents/skewl/PhotoNet/ScatteringNet/ScatteringNet_Matlab/data/CompleteDataFiles/6_layer_tio2_combined_06_21')
    parser.add_argument("--data",type=str,default='data/CompleteDataFiles/8_layer_tio2_combined')
    parser.add_argument("--reuse_weights",type=str,default='True')
    parser.add_argument("--output_folder",type=str,default='results/begin_test_6_Layer_TiO2_225_layer/')
        #Generate the loss file/val file name by looking to see if there is a previous one, then creating/running it.
    parser.add_argument("--weight_name_load",type=str,default="")#This would be something that goes infront of w_1.txt. This would be used in saving the weights
    parser.add_argument("--weight_name_save",type=str,default="")
    parser.add_argument("--n_batch",type=int,default=100)
    parser.add_argument("--numEpochs",type=int,default=200)
    parser.add_argument("--lr_rate",default=.001)
    parser.add_argument("--lr_decay",default=.99)
    parser.add_argument("--num_layers",default=4)
    parser.add_argument("--n_hidden",default=250)
    parser.add_argument("--percent_val",default=.2)

    args = parser.parse_args()
    dict = vars(args)

    for i in dict:
        if (dict[i]=="False"):
            dict[i] = False
        elif dict[i]=="True":
            dict[i] = True
        
    kwargs = {  
            'data':dict['data'],
            'reuse_weights':dict['reuse_weights'],
            'output_folder':dict['output_folder'],
            'weight_name_save':dict['weight_name_save'],
            'weight_name_load':dict['weight_name_load'],
            'n_batch':dict['n_batch'],
            'numEpochs':dict['numEpochs'],
            'lr_rate':dict['lr_rate'],
            'lr_decay':dict['lr_decay'],
            'num_layers':dict['num_layers'],
            'n_hidden':dict['n_hidden'],
            'percent_val':dict['percent_val']
            }

    main(**kwargs)




