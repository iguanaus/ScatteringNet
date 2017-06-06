import tensorflow as tf
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import os
import time
#from numpy import genfromtxt


RANDOM_SEED = 42
tf.set_random_seed(RANDOM_SEED)


cum_loss_file = "results/Dielectric_Four/temp_loss.txt"
resuse_weights = True
data_test_file= "data/double_dielectrics.csv"
data_train_file= "data/double_dielectrics_val.csv"
n_batch = 1
numEpochs=50000
output_weights_folder = "results/Dielectric_Four/"
lr_rate = 0.00005
lr_decay = 0.9
weightDir = 'results/Dielectric_Four/'

#This is the position in the list to sample.
spects_to_sample = [30000]


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
    print("My x size: " , x_size)
    h_size = 20                # Number of hidden nodes
    y_size = train_Y.shape[1]   # Number of outcomes (3 iris flowers)

    # Symbols
    X = tf.placeholder("float", shape=[None, x_size])
    y = tf.placeholder("float", shape=[None, y_size])

    # Weight initializations
    if resuse_weights:
        weight_1 = np.loadtxt(weightDir + "w_1.txt",delimiter=',')
        #weight_1 = np.array(np.loadtxt("results/FixedTwoNetwork/w_1.txt",delimiter=','))
        print("Weight 1: " , weight_1)
        weight_2 = np.loadtxt(weightDir +"w_2.txt",delimiter=',')
        #print("Weight 2: " , weight_2)
        weight_3 = np.loadtxt(weightDir +"w_3.txt",delimiter=',')
        weight_4 = np.loadtxt(weightDir +"w_4.txt",delimiter=',')
        weight_5 = np.loadtxt(weightDir +"w_5.txt",delimiter=',')
        weight_6 = np.loadtxt(weightDir +"w_6.txt",delimiter=',')
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
    print("My X Values: ")
    print("X  :" , X)
    print("w_1:" , w_1)
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

        start_time=time.time()
        print("========                         Iterations started                  ========")
        



        #print("Train X: " , train_X)
        for ele in spects_to_sample:
            step = ele

            batch_x = train_X[step * n_batch : (step+1) * n_batch]
            batch_y = train_Y[step * n_batch : (step+1) * n_batch]



            print("Input: " , batch_x[0])
            print("Desired out: " , batch_y)
            myvals0 = sess.run(yhat,feed_dict={X:batch_x,y:batch_y})
            print("NN out: " , myvals0)
            #I need to write these to a file.
            filename = 'test_out_file_'+str(ele)+'.txt'
            f = open(filename,'w')
            f.write("XValue\nActual\nPredicted\n")
            f.write(str(batch_x[0])+"\n")
            for item in list(batch_y[0]):
                f.write(str(item) + ",")
            f.write("\n")
            for item in list(myvals0[0]):
                f.write(str(item) + ",")
            f.write("\n")
            f.flush()
            f.close()
            print("Wrote to: " + str(filename))

            

    print "========Iterations completed in : " + str(time.time()-start_time) + " ========"
        
    sess.close()

if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description="Physics Net Training")
    parser.add_argument("--data",type=str,default='data/dielectric_spectrums_four')
    parser.add_argument("--output_folder",type=str,default='results/Dielectric_Four/')
        #Generate the loss file/val file name by looking to see if there is a previous one, then creating/running it.
    parser.add_argument("--weight_name_load",type=str,default="")#This would be something that goes infront of w_1.txt. This would be used in saving the weights
    parser.add_argument("--spect_to_sample",type=int,default=10000)
    parser.add_argument("--sample_val",type=str,default="True")
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
            'output_folder':dict['output_folder'],
            'weight_name_load':dict['weight_name_load'],
            'spects_to_sample':dict['spects_to_sample'],
            'sample_val':dict['sample_val'],
            'num_layers':dict['num_layers'],
            'n_hidden':dict['n_hidden'],
            'percent_val':dict['percent_val']
            }

    main(**kwargs)
