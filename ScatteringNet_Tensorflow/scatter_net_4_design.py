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
cum_loss_file = "short_fixed_net_5x20.txt"
resuse_weights = True
data_test_file= "data/test_answer.csv"
data_train_file= "results/Dielectric_Four/test_dielectric_large_45_15_40_15_val.csv"
init_list =[50,50,50,50]
#[[ 20.22009659  20.06668854  20.03667641  20.72088623]]
numInput = len(init_list)

def init_weights(shape):
    """ Weight initialization """
    weights = tf.random_normal(shape, stddev=.1)
    return tf.Variable(weights)

def forwardprop(X, w_1, w_2,w_3,w_4,w_5,w_6):
    """
    Forward-propagation.
    IMPORTANT: yhat is not softmax since TensorFlow's softmax_cross_entropy_with_logits() does that internally.
    """

    #The X's I want to take the max at. 
    X = tf.maximum(X,10)#max(X,10) #Make sure it is above 10
    X = tf.minimum(X,100) #Make sure it is below 40.

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

def get_data(test_file="test.csv",file_val="test_val.csv"):
    #trainX = np.array([])
    #For single
    #train_X = np.transpose(np.genfromtxt('test_val.csv', delimiter=',')),(-1,1))
    #For Multi
    train_X = np.genfromtxt(data_train_file, delimiter=',')

    train_Y = np.transpose(np.genfromtxt(data_test_file, delimiter=','))


    print(train_Y)



    indices = np.random.permutation(train_X.shape[0]) #This gives us the ordering

    #print("Sample of x: " , train_X[0])
    #print("Sample of y: " , train_Y[0])
    new_train_X = []
    new_train_Y = []
    for ele in indices:
        #For multiple
        new_train_X.append(list(train_X[ele]))
        #For single
        #new_train_X.append(list(train_X[ele][0]))
        new_train_Y.append(list(train_Y[ele]))
    
    #print("New train X: " , new_train_X)
    #print("New train Y: " , new_train_Y)

    new_train_X = np.array(new_train_X)
    new_train_Y = np.array(new_train_Y)

    #print("Final New train X: " , new_train_X)
    #print("Final New train Y: " , new_train_Y)




    print("X shape: " , new_train_X.shape)
    print("Y shape: " , new_train_Y.shape)

    #train_X = np.array([[1,1,1,1],[2,2,2,2],[3,3,3,3],[4,4,4,4],[5,5,5,5]])
    #train_Y = np.array([[2,2],[3,3],[4,4],[5,5],[6,6]])
    return new_train_X, new_train_Y 

def gen_data_first(test_file="test.csv"):
    #train_X = np.reshape(np.transpose(np.genfromtxt(data_train_file, delimiter=',')),(-1,1))
    train_X = np.array([np.genfromtxt(data_train_file, delimiter=',')])
    train_Y = np.array([np.genfromtxt(data_test_file, delimiter=',')]) #37
    print(train_X,train_Y)
    return train_X, train_Y


def main():
    #train_X, train_Y = get_data()
    train_X, train_Y = gen_data_first()

    #print("Train ")
    print("It should be: " , train_X)

    #print("Train_X: " , train_X)
    #os.exit()

    #train_X, test_X, train_y, test_y = get_iris_data()

    # Layer's sizes
    print("Train x shape is: " , train_X.shape)
    x_size = train_X.shape[1]   # Number of input nodes: 4 features and 1 bias
    h_size = 20                # Number of hidden nodes
    y_size = train_Y.shape[1]   # Number of outcomes (3 iris flowers)

    # Symbols
    #X = tf.placeholder("float", shape=[None, x_size])
    #X = tf.Variable()

    X = tf.get_variable(name="b1", shape=[1,numInput], initializer=tf.constant_initializer(init_list))

    #print("X: " , X)

    y = tf.placeholder("float", shape=[None, y_size])

    # Weight initializations
    if resuse_weights:
        #weight_1 = np.array([np.loadtxt("results/Dielectric_Four/w_1.txt",delimiter=',')])[0]
        weight_1 = np.loadtxt("results/Dielectric_Four/w_1.txt",delimiter=",")
        #print("Weight 1: " , weight_1)
        #print("Weight 1: " , weight_1)
        weight_2 = np.loadtxt("results/Dielectric_Four/w_2.txt",delimiter=',')
        #print("Weight 2: " , weight_2)
        weight_3 = np.loadtxt("results/Dielectric_Four/w_3.txt",delimiter=',')
        weight_4 = np.loadtxt("results/Dielectric_Four/w_4.txt",delimiter=',')
        weight_5 = np.loadtxt("results/Dielectric_Four/w_5.txt",delimiter=',')
        weight_6 = np.loadtxt("results/Dielectric_Four/w_6.txt",delimiter=',')
        w_1 = tf.Variable(weight_1,dtype=tf.float32)
        #print(w_1)
        w_2 = tf.Variable(weight_2,dtype=tf.float32)
        w_3 = tf.Variable(weight_3,dtype=tf.float32)
        w_4 = tf.Variable(weight_4,dtype=tf.float32)
        w_5 = tf.Variable(weight_5,dtype=tf.float32)
        w_6 = tf.Variable(weight_6,dtype=tf.float32)
        #os.exit()
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

    #Oh wait, the easy way to do this is have the y be '1' for those that you want.

    topval = tf.abs(tf.matmul(y,tf.transpose(tf.abs(yhat)))) #This will select all the values that we want.

    botval = tf.abs(tf.matmul(tf.abs(y-1),tf.transpose(tf.abs(yhat)))) #This will get the values that we do not want. 


    #topval = tf.mean(tf.multiply(yhay,y))#This would select only the ones that we want

    #botval = tf.mean(tf.multiply(yhat,tf.abs(y-1)))#THis would select the opposte batch

    #cost = 1.0/topval
    cost = botval/topval#topval#/botval

    #Then we just need to feed to y the ones that we want to maximize.

    
    #Output float values)
    #cost    = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=yhat))
    optimizer = tf.train.RMSPropOptimizer(learning_rate=0.0005, decay=0.9).minimize(cost,var_list=[X])
    #updates = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

    # Run SGD
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        n_batch = 1

        n_iter = 10000000
        step = 0

        numEpochs=1000

        curEpoch=0
        #print("Train x shape: " , train_X.shape)
        cum_loss = 0
        f2 = open(cum_loss_file,'w')

        start_time=time.time()
        print("========                         Iterations started                  ========")

        #print("Train y: " , train_Y)

        while curEpoch < numEpochs:

            #batch_x = train_X[step * n_batch : (step+1) * n_batch]
            batch_y = train_Y#[step * n_batch : (step+1) * n_batch]
            #print("BatchY: ", train_Y)
            sess.run(optimizer, feed_dict={y: batch_y})
            loss = sess.run(cost,feed_dict={y:batch_y})
            cum_loss += loss        
            step += 1
            #print("Step: " , step)
            #print("Loss: " , loss)
            if step == 100:
                step = 0
                curEpoch +=1            
                f2.write(str(float(cum_loss))+str("\n"))
                if (curEpoch % 100 == 0 or curEpoch == 1):
                    myvals0 = sess.run(yhat,feed_dict={y:batch_y})
                    print("Epoch: " + str(curEpoch+1) + " : Loss: " + str(cum_loss))
                    print(list((myvals0)[0]))
                    print(X.eval())
                cum_loss = 0
        #print(w_1)
        weight_1 = w_1.eval()
        weight_2 = w_2.eval()
        weight_3 = w_3.eval()
        weight_4 = w_4.eval()
        weight_5 = w_5.eval()
        weight_6 = w_6.eval()
        print(weight_1)
        print(np.array(weight_1))
        print(X.eval())
        # np.savetxt("results/w_1.txt",weight_1,delimiter=',')
        # np.savetxt("results/w_2.txt",weight_2,delimiter=',')
        # np.savetxt("results/w_3.txt",weight_3,delimiter=',')
        # np.savetxt("results/w_4.txt",weight_4,delimiter=',')
        # np.savetxt("results/w_5.txt",weight_5,delimiter=',')
        # np.savetxt("results/w_6.txt",weight_6,delimiter=',')




    print "========Iterations completed in : " + str(time.time()-start_time) + " ========"
        
    sess.close()

if __name__ == '__main__':
    main()