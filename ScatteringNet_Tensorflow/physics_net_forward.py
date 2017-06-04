# Implementation of a simple MLP network with one hidden layer. Tested on the iris data set.
# Requires: numpy, sklearn>=0.18.1, tensorflow>=1.0

# NOTE: In order to make the code simple, we rewrite x * W_1 + b_1 = x' * W_1'
# where x' = [x | 1] and W_1' is the matrix W_1 appended with a new row with elements b_1's.
# Similarly, for h * W_2 + b_2
import tensorflow as tf
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import os
import time
#from numpy import genfromtxt


RANDOM_SEED = 42
tf.set_random_seed(RANDOM_SEED)
cum_loss_file = "dieletric_loss.csv"
val_loss_file = "dielectric_loss_val.csv"
resuse_weights = False
data_test_file= "data/dielectric_spectrums_four.csv"
data_train_file= "data/dielectric_spectrums_four_val.csv"
#data_test_file2= "data/dielectric_spectrums_power_two.csv"
#data_train_file2= "data/dielectric_spectrums_power_two_val.csv"
output_weights_folder = "results/Dielectric_Four/"
n_batch = 100
numEpochs=5000
lr_rate = 0.0005
lr_decay = 0.9


def init_weights(shape):
    """ Weight initialization """
    weights = tf.random_normal(shape, stddev=.1)
    return tf.Variable(weights)

def forwardprop(X, w_1, w_2,w_3,w_4,w_5,w_6):
    """
    Forward-propagation.
    IMPORTANT: yhat is not softmax since TensorFlow's softmax_cross_entropy_with_logits() does that internally.
    """

    h    = tf.nn.sigmoid(tf.matmul(X, w_1))  # The \sigma function
    h1    = tf.nn.sigmoid(tf.matmul(h, w_3))  # The \sigma function
    h2    = tf.nn.sigmoid(tf.matmul(h1, w_4))
    h3    = tf.nn.sigmoid(tf.matmul(h2, w_5))
    h4    = tf.nn.sigmoid(tf.matmul(h3, w_6))
    
    yhat = tf.matmul(h4, w_2)  # The \varphi function
    return yhat

#This method reads from the 'X' and 'Y' file and gives in the input as an array of arrays (aka if the input dim is 5 and there are 10 training sets, the input is a 10X 5 array)
#a a a a a       3 3 3 3 3 
#b b b b b       4 4 4 4 4
#c c c c c       5 5 5 5 5

def get_data(test_file="data/test.csv",file_val="data/test_val.csv",percentTest=0.2):

    train_X = np.genfromtxt(file_val, delimiter=',')
    train_Y = np.transpose(np.genfromtxt(test_file,delimiter=','))

    X_train, X_val, y_train, y_val = train_test_split(train_X,train_Y,test_size=percentTest,random_state=42)

    return X_train, y_train, X_val, y_val



def main():
    train_X, train_Y , val_X, val_Y = get_data(data_test_file,data_train_file)
    #print("Train_X: " , train_X)
    #os.exit()

    #train_X, test_X, train_y, test_y = get_iris_data()

    # Layer's sizes
    x_size = train_X.shape[1]   # Number of input nodes: 4 features and 1 bias
    h_size = 20                # Number of hidden nodes
    y_size = train_Y.shape[1]   # Number of outcomes (3 iris flowers)

    # Symbols
    X = tf.placeholder("float", shape=[None, x_size])
    y = tf.placeholder("float", shape=[None, y_size])

    # Weight initializations
    if resuse_weights:
        weight_1 = np.array([np.loadtxt("results/w_1_large.txt",delimiter=',')])
        #print("Weight 1: " , weight_1)
        weight_2 = np.loadtxt("results/w_2_large.txt",delimiter=',')
        #print("Weight 2: " , weight_2)
        weight_3 = np.loadtxt("results/w_3_large.txt",delimiter=',')
        weight_4 = np.loadtxt("results/w_4_large.txt",delimiter=',')
        weight_5 = np.loadtxt("results/w_5_large.txt",delimiter=',')
        weight_6 = np.loadtxt("results/w_6_large.txt",delimiter=',')
        w_1 = tf.Variable(weight_1,dtype=tf.float32)
        #print(w_1)
        w_2 = tf.Variable(weight_2,dtype=tf.float32)
        w_3 = tf.Variable(weight_3,dtype=tf.float32)
        w_4 = tf.Variable(weight_4,dtype=tf.float32)
        w_5 = tf.Variable(weight_5,dtype=tf.float32)
        w_6 = tf.Variable(weight_6,dtype=tf.float32)

    else:
        w_1 = init_weights((x_size, h_size))
        #print(w_1)

        w_3 = init_weights((h_size, h_size))
        w_4 = init_weights((h_size, h_size))
        w_5 = init_weights((h_size, h_size))
        w_6 = init_weights((h_size, h_size))

        w_2 = init_weights((h_size, y_size))

    # Forward propagation
    yhat    = forwardprop(X, w_1, w_2,w_3,w_4,w_5,w_6)
    
    # Backward propagation
    cost = tf.reduce_sum(tf.square(y-yhat))
    #Output float values)
    #cost    = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=yhat))
    optimizer = tf.train.RMSPropOptimizer(learning_rate=lr_rate, decay=lr_decay).minimize(cost)
    #updates = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

    # Run SGD
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        n_iter = 10000000
        step = 0


        curEpoch=0
        #print("Train x shape: " , train_X.shape)
        cum_loss = 0
        f2 = open(cum_loss_file,'w')
        f3 = open(val_loss_file,'w')

        start_time=time.time()
        print("========                         Iterations started                  ========")


        #print("Train X: " , train_X)
        while curEpoch < numEpochs:

            batch_x = train_X[step * n_batch : (step+1) * n_batch]
            #print("Batch X: " , batch_x)
            batch_y = train_Y[step * n_batch : (step+1) * n_batch]
            sess.run(optimizer, feed_dict={X: batch_x, y: batch_y})
            loss = sess.run(cost,feed_dict={X:batch_x,y:batch_y})
            cum_loss += loss        
            step += 1
            if step == int(train_X.shape[0]/n_batch):
                print("Loss: " , loss)
                step = 0
                curEpoch +=1            
                f2.write(str(float(cum_loss))+str("\n"))
                if (curEpoch % 10 == 0 or curEpoch == 1):
                    #Calculate the validation loss

                    val_loss = sess.run(cost,feed_dict={X:val_X,y:val_Y})
                    print("Validation loss: " , str(val_loss))
                    f3.write(str(float(val_loss))+str("\n"))
                    f3.flush()

                    myvals0 = sess.run(yhat,feed_dict={X:batch_x,y:batch_y})
                    print("Epoch: " + str(curEpoch+1) + " : Loss: " + str(cum_loss))
                    myvals0 = sess.run(yhat,feed_dict={X:train_X[0:1],y:train_Y[0:1]})
                    #print("Myvals0:",myvals0)
                    #print("Batch y: " , train_Y[0:1])
                    #print("Residuals:", myvals0-train_Y[0:1])
                    myvals0 = myvals0[0]
                    #myvals1 = sess.run(yhat,feed_dict={X:train_X[-180:-179],y:train_Y[-180:-179]})[0] #Large dim inputs
                    myvals2 = sess.run(yhat,feed_dict={X:train_X[-2:-1],y:train_Y[-2:-1]})[0]
                    f2.flush()
                cum_loss = 0
        #print(w_1)
        weight_1 = w_1.eval()
        weight_2 = w_2.eval()
        weight_3 = w_3.eval()
        weight_4 = w_4.eval()
        weight_5 = w_5.eval()
        weight_6 = w_6.eval()
        #print(weight_1)
        #print(np.array(weight_1))
        np.savetxt(output_weights_folder +"w_1.txt",weight_1,delimiter=',')
        np.savetxt(output_weights_folder +"w_2.txt",weight_2,delimiter=',')
        np.savetxt(output_weights_folder +"w_3.txt",weight_3,delimiter=',')
        np.savetxt(output_weights_folder +"w_4.txt",weight_4,delimiter=',')
        np.savetxt(output_weights_folder +"w_5.txt",weight_5,delimiter=',')
        np.savetxt(output_weights_folder +"w_6.txt",weight_6,delimiter=',')




    print "========Iterations completed in : " + str(time.time()-start_time) + " ========"
        
    sess.close()

if __name__ == '__main__':
    main()