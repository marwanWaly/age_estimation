def newAges(ages, age, i):
  if age < 5:
    ages[i][0] = 1
    return ages  
  if age < 10:
    ages[i][1] = 1
    return ages 
  if age < 15:
    ages[i][2] = 1
    return ages
  if age < 20:
    ages[i][3] = 1
    return ages
  if age < 30:
    ages[i][4] = 1
    return ages
  if age < 40:
    ages[i][5] = 1
    return ages
  if age < 50:
    ages[i][6] = 1
    return ages
  if age < 60:
    ages[i][7] = 1
    return ages
  if age < 70:
    ages[i][8] = 1
    return ages
  if age < 80:
    ages[i][9] = 1
    return ages
  ages[i][10] = 1
  return ages

def newAges2(ages, age, i):
  if age < 20:
    ages[i][0] = 1
    return ages  
  if age < 60:
    ages[i][1] = 1
    return ages 
  ages[i][2] = 1
  return ages



import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
import cv2
import pickle

path = "/content/UTKFace"
n_classes = 11
batch_size = 128
scale = 64
numOfChannel = 1
numberOfImg = len(os.listdir(path))


data = []
ages = np.zeros(numberOfImg * n_classes).reshape((numberOfImg, n_classes))
i = 0
for img in os.listdir(path):
    img_mat = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
    age = img.split('_')[0]
    dim = (scale, scale)
    img_mat = cv2.resize(img_mat, dim)
    data.append(img_mat)
    ages = newAges(ages, int(age), i)
    i = i+1
    

data = np.array(data) 

!touch data.pickle
pk_out = open("data.pickle","wb")
pickle.dump(data,pk_out)
pk_out.close()

!touch ages.pickle
pk_out = open("ages.pickle","wb")
pickle.dump(ages,pk_out)
pk_out.close()

pickle_in = open("data.pickle","rb")
X1 = pickle.load(pickle_in)

pickle_in = open("ages.pickle","rb")
y1 = pickle.load(pickle_in)

X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=0.05, random_state=42)

X_train = X_train.reshape(-1, scale, scale, numOfChannel)
y_train = y_train.reshape(-1, n_classes)

X_test = X_test.reshape(-1, scale, scale, numOfChannel)
y_test = y_test.reshape(-1, n_classes)




x = tf.placeholder('float', [None, scale,scale,numOfChannel])
y = tf.placeholder('float', [None, n_classes])


def conv2d(x,W,b,strides=1):
    x = tf.nn.conv2d(x,W,strides=[1,strides,strides,1],padding="SAME")
    x = tf.nn.bias_add(x,b)
    return tf.nn.relu(x)


def maxpool2d(x):
    return tf.nn.max_pool(x , ksize=[1,2,2,1] , strides = [1,2,2,1] , padding = 'SAME')


def Convolutional_neural_network(x):
    weights = {'W_conv1':tf.Variable(tf.random_normal([5,5,numOfChannel,32])),
               'W_conv2':tf.Variable(tf.random_normal([5,5,32,64])),
                'W_fc':tf.Variable(tf.random_normal([16*16*64 , 1024])),
                'out':tf.Variable(tf.random_normal([1024 , n_classes]))
              }

    biases = {'b_conv1':tf.Variable(tf.random_normal([32])),
                'b_conv2':tf.Variable(tf.random_normal([64])),
                'b_fc':tf.Variable(tf.random_normal([1024])),
                'out':tf.Variable(tf.random_normal([n_classes]))
              }


    x = tf.reshape(x,shape = [-1,scale,scale,numOfChannel])

    conv1 = conv2d(x,weights['W_conv1'], biases['b_conv1'])
    conv1 = maxpool2d(conv1)

    conv2 = conv2d(conv1,weights['W_conv2'], biases['b_conv2'])
    conv2 = maxpool2d(conv2)
     
    fc = tf.reshape(conv2,[-1,16*16*64])
    fc = tf.nn.relu(tf.matmul(fc,weights['W_fc']) + biases['b_fc'])
    output = tf.matmul(fc,weights['out'])  + biases['out']

    return output

def train_neural_network(x):
    prediction = Convolutional_neural_network(x)
    # OLD VERSION:
    #cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction,y) )
    # NEW:
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    hm_epochs = 50
    with tf.Session() as sess:
        # OLD:
        #sess.run(tf.initialize_all_variables())
        # NEW:
        sess.run(tf.global_variables_initializer())
        for epoch in range(hm_epochs):
            epoch_loss = 0
            for batch in range(len(X_train)//batch_size):
                epoch_x = X_train[batch*batch_size:min((batch+1)*batch_size, len(X_train))]
                epoch_y = y_train[batch*batch_size:min((batch+1)*batch_size, len(y_train))]
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c
               
                
            
            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
            save = correct
            X_t = X_test
            y_t = y_test
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            print('Epoch', epoch+1, 'completed out of',hm_epochs,'loss:',epoch_loss,'   Accuracy:' ,accuracy.eval({x:X_t, y:y_t}))
            

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        save = correct
        X_t = X_test
        y_t = y_test
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:',accuracy.eval({x:X_t, y:y_t}))
        sess.close()
        return save
        
        

mmb = train_neural_network(x)