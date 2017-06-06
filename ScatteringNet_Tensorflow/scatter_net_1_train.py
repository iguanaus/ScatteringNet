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

RANDOM_SEED = 42
tf.set_random_seed(RANDOM_SEED)

def init_weights(shape):
    """ Weight initialization """
    weights = tf.random_normal(shape, stddev=.1)
    return tf.Variable(weights)

def save_weights(weights,output_folder,weight_name_save,num_layers):
    for i in xrange(0, num_layers):
        weight_i = weights[i].eval()
        np.savetxt(output_folder+weight_name_save+"w_"+str(i)+".txt",weight_i,delimiter=',')
    return

def load_weights(output_folder,weight_load_name,num_layers):
    weights = []
    for i in xrange(0, num_layers):
        weight_i = np.loadtxt(output_folder+weight_load_name+"w_"+str(i)+".txt")
        w_i = tf.Variable(weight_i,dtype=tf.float32)
        weights.append(w_i)
    return weights

def forwardprop(X, weights,num_layers):
    htemp = None
    for i in xrange(0, num_layers):
        if i ==0:
            htemp = tf.nn.sigmoid(tf.matmul(X,weights[i]))    
        else:   
            htemp = tf.nn.sigmoid(tf.matmul(htemp,weights[i]))
    yval = tf.matmul(htemp,weights[-1])
    return yval

#This method reads from the 'X' and 'Y' file and gives in the input as an array of arrays (aka if the input dim is 5 and there are 10 training sets, the input is a 10X 5 array)
#a a a a a       3 3 3 3 3 
#b b b b b       4 4 4 4 4
#c c c c c       5 5 5 5 5

def get_data(data,percentTest=.2,random_state=42):
    x_file = data+"_val.csv"
    y_file = data+".csv"
    train_X = np.genfromtxt(x_file,delimiter=',')
    train_Y = np.transpose(np.genfromtxt(y_file,delimiter=','))
    X_train, X_val, y_train, y_val = train_test_split(train_X,train_Y,test_size=percentTest,random_state=random_state)
    return X_train, y_train, X_val, y_val

def main(data,reuse_weights,output_folder,weight_name_save,weight_name_load,n_batch,numEpochs,lr_rate,lr_decay,num_layers,n_hidden,percent_val):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    train_X, train_Y , val_X, val_Y = get_data(data,percentTest=percent_val)

    x_size = train_X.shape[1]   # Number of input nodes: 4 features and 1 bias
    y_size = train_Y.shape[1]   # Number of outcomes (3 iris flowers)

    # Symbols
    X = tf.placeholder("float", shape=[None, x_size])
    y = tf.placeholder("float", shape=[None, y_size])
    weights = []

    # Weight initializations
    if reuse_weights:
        weights = load_weights(output_folder,weight_name_load,num_layers)

    else:
        for i in xrange(0,num_layers):
            if i ==0:
                weights.append(init_weights((x_size,n_hidden)))
            else:
                weights.append(init_weights((n_hidden,n_hidden)))
        weights.append(init_weights((n_hidden,y_size)))
    # Forward propagation
    yhat    = forwardprop(X, weights,num_layers)
    
    # Backward propagation
    cost = tf.reduce_sum(tf.square(y-yhat))
    optimizer = tf.train.RMSPropOptimizer(learning_rate=lr_rate, decay=lr_decay).minimize(cost)

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
        val_loss_file = open(output_folder+"train_val_loss_"+str(numFile) + "_val.txt",'w')
        start_time=time.time()
        print("========                         Iterations started                  ========")
        while curEpoch < numEpochs:
            batch_x = train_X[step * n_batch : (step+1) * n_batch]
            batch_y = train_Y[step * n_batch : (step+1) * n_batch]
            sess.run(optimizer, feed_dict={X: batch_x, y: batch_y})
            cum_loss += sess.run(cost,feed_dict={X:batch_x,y:batch_y})
            step += 1
            if step == int(train_X.shape[0]/n_batch): #Epoch finished
                step = 0
                curEpoch +=1            
                train_loss_file.write(str(float(cum_loss))+str("\n"))
                if (curEpoch % 10 == 0 or curEpoch == 1):
                    #Calculate the validation loss
                    val_loss = sess.run(cost,feed_dict={X:val_X,y:val_Y})
                    print("Validation loss: " , str(val_loss))
                    val_loss_file.write(str(float(val_loss))+str("\n"))
                    val_loss_file.flush()

                    print("Epoch: " + str(curEpoch+1) + " : Loss: " + str(cum_loss))
                    train_loss_file.flush()
                cum_loss = 0
        save_weights(weights,output_folder,weight_name_save,num_layers)
    print "========Iterations completed in : " + str(time.time()-start_time) + " ========"
    sess.close()

if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description="Physics Net Training")
    parser.add_argument("--data",type=str,default='data/dielectric_spectrums_four')
    parser.add_argument("--reuse_weights",type=str,default='False')
    parser.add_argument("--output_folder",type=str,default='results/Dielectric_Four/')
        #Generate the loss file/val file name by looking to see if there is a previous one, then creating/running it.
    parser.add_argument("--weight_name_load",type=str,default="")#This would be something that goes infront of w_1.txt. This would be used in saving the weights
    parser.add_argument("--weight_name_save",type=str,default="")
    parser.add_argument("--n_batch",type=int,default=100)
    parser.add_argument("--numEpochs",type=int,default=5000)
    parser.add_argument("--lr_rate",default=.0005)
    parser.add_argument("--lr_decay",default=.9)
    parser.add_argument("--num_layers",default=4)
    parser.add_argument("--n_hidden",default=30)
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



