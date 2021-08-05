import tensorflow as tf 
import numpy
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.contrib.layers import flatten
from sklearn.utils import shuffle
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
from sklearn.model_selection import train_test_split

import IPython
# tf.set_random_seed(0)
df_train = pd.read_csv('dataset/train.csv')
df_test = pd.read_csv('dataset/test.csv')
df_train.head()

df_train = pd.get_dummies(df_train,columns=['label'])
df_features = df_train.iloc[:, :-10].values
df_label = df_train.iloc[:, -10:].values
print(df_features.shape)

X_train, X_test, y_train, y_test = train_test_split(df_features, df_label, test_size = 0.2, random_state = 1212)
#X_test, X_validation, y_test,y_validation = train_test_split(X_test,  y_test, test_size=0.5, random_state=0)
image_size = 28
num_labels = 10
num_channels = 1 # grayscale
def reformat(dataset, labels):
    dataset = dataset.reshape((-1, image_size, image_size, num_channels)).astype(np.float32)
    labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
    return dataset, labels

train_dataset, train_labels = reformat(X_train, y_train)

#valid_dataset, valid_labels = reformat(X_validation, y_validation)
test_dataset , test_labels = reformat(X_test, y_test)
df_test = df_test.to_numpy().reshape((-1, image_size, image_size, num_channels)).astype(np.float32)
print ('Training set :', train_dataset.shape, train_labels.shape)
#print ('Validation set :', valid_dataset.shape, valid_labels.shape)
print ('Test set :', test_dataset.shape, test_labels.shape)

# Pad images with 0s
X_train      = np.pad(train_dataset, ((0,0),(2,2),(2,2),(0,0)), 'constant')
#X_validation = np.pad(valid_dataset, ((0,0),(2,2),(2,2),(0,0)), 'constant')
X_test       = np.pad(test_dataset, ((0,0),(2,2),(2,2),(0,0)), 'constant')
    
print ('Training set after padding 2x2    :', X_train.shape, train_labels.shape)
#print ('Validation set after padding 2x2  :', X_validation.shape, valid_labels.shape)
print ('Test set after padding 2x2        :', X_test.shape, test_labels.shape)
print ('Submission data after padding 2x2 :', X_test.shape)

x = tf.placeholder(tf.float32, shape=[None,32,32,1])
y_ = tf.placeholder(tf.int32, (None))
# LeNet-5 architecture implementation using TensorFlow
# def LeNet_5(x):
#IPython.embed()
# Layer 1 : Convolutional Layer. Input = 32x32x1, Output = 28x28x1.
conv1_w = tf.Variable(tf.truncated_normal(shape = [5,5,1,6],mean = 0, stddev = 0.1))
conv1_b = tf.Variable(tf.zeros(6))
conv1 = tf.nn.conv2d(x,conv1_w, strides = [1,1,1,1], padding = 'VALID') + conv1_b 
# TODO: Activation.
conv1 = tf.nn.relu(conv1)

# Pooling Layer. Input = 28x28x1. Output = 14x14x6.
pool_1 = tf.nn.max_pool(conv1, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID')


# TODO: Layer 2: Convolutional. Output = 10x10x16.
conv2_w = tf.Variable(tf.truncated_normal(shape = [5,5,6,16], mean = 0, stddev = 0.1))
conv2_b = tf.Variable(tf.zeros(16))
conv2 = tf.nn.conv2d(pool_1, conv2_w, strides = [1,1,1,1], padding = 'VALID') + conv2_b
# TODO: Activation.
conv2 = tf.nn.relu(conv2)# TODO: Pooling. Input = 10x10x16. Output = 5x5x16.
pool_2 = tf.nn.max_pool(conv2, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID')


# TODO: Flatten. Input = 5x5x16. Output = 400.
fc1 = flatten(pool_2)


# TODO: Layer 3: Fully Connected. Input = 400. Output = 120.
fc1_w = tf.Variable(tf.truncated_normal(shape = (400,120), mean = 0, stddev = 0.1))
fc1_b = tf.Variable(tf.zeros(120))
fc1 = tf.matmul(fc1,fc1_w) + fc1_b

# TODO: Activation.
fc1 = tf.nn.relu(fc1)

# TODO: Layer 4: Fully Connected. Input = 120. Output = 84.
fc2_w = tf.Variable(tf.truncated_normal(shape = (120,84), mean = 0, stddev = 0.1))
fc2_b = tf.Variable(tf.zeros(84))
fc2 = tf.matmul(fc1,fc2_w) + fc2_b
# TODO: Activation.
fc2 = tf.nn.relu(fc2)

# TODO: Layer 5: Fully Connected. Input = 84. Output = 10.
fc3_w = tf.Variable(tf.truncated_normal(shape = (84,10), mean = 0 , stddev = 0.1))
fc3_b = tf.Variable(tf.zeros(10))
logits = tf.matmul(fc2, fc3_w) + fc3_b




#Invoke LeNet function by passing features
#logits,   conv1_w, conv1_b, conv2_w, conv2_b, fc1_w, fc1_b, fc2_w, fc2_b, fc3_w, fc3_b, fc1, fc2 = LeNet_5(x)#Softmax with cost function implementation
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels = y_, logits = logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = 0.001)
training_operation = optimizer.minimize(loss_operation)


correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(y_,1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))# Evaluate function
def evaluate(X_data, y_data):
     num_examples = len(X_data)
     total_accuracy = 0
     sess = tf.get_default_session()
     for offset in range(0, num_examples, BATCH_SIZE):
         batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
         accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y_: batch_y})
         total_accuracy += (accuracy * len(batch_x))
     return total_accuracy / num_examples



#To initialise session and run
EPOCHS = 40
BATCH_SIZE = 128
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    
    print("Training... with dataset - ", num_examples)
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y_: batch_y})
            
        #validation_accuracy = evaluate(X_validation, y_validation)
        print("EPOCH {} ...".format(i+1))
        train_accuracy = evaluate(X_train, y_train)
        print("Train Accuracy = {:.3f}".format(train_accuracy))
        #print()
        
        
    
        test_accuracy = evaluate(X_test, y_test)
        print("Test Accuracy = {:.3f}".format(test_accuracy))
    # save 
    saver = tf.train.Saver()
    save_path = saver.save(sess, 'lenet.ckpt')
    print("Model saved %s "%save_path)
    np.savetxt('data/conv1_w', sess.run(conv1_w, feed_dict={x: X_train, y_: y_train}).flatten(),  delimiter=',')
    np.savetxt('data/conv1_b', sess.run(conv1_b, feed_dict={x: X_train, y_: y_train}),  delimiter=',')
    np.savetxt('data/conv2_w', sess.run(conv2_w, feed_dict={x: X_train, y_: y_train}).flatten(),  delimiter=',')
    np.savetxt('data/conv2_b', sess.run(conv2_b, feed_dict={x: X_train, y_: y_train}),  delimiter=',')
    np.savetxt('data/fc1_w', sess.run(fc1_w, feed_dict={x: X_train, y_: y_train}),  delimiter=',')
    np.savetxt('data/fc1_b', sess.run(fc1_b, feed_dict={x: X_train, y_: y_train}),  delimiter=',')
    np.savetxt('data/fc2_w', sess.run(fc2_w, feed_dict={x: X_train, y_: y_train}),  delimiter=',')
    np.savetxt('data/fc2_b', sess.run(fc2_b, feed_dict={x: X_train, y_: y_train}),  delimiter=',')
    np.savetxt('data/fc3_w', sess.run(fc3_w, feed_dict={x: X_train, y_: y_train}),  delimiter=',')
    np.savetxt('data/fc3_b', sess.run(fc3_b, feed_dict={x: X_train, y_: y_train}),  delimiter=',')
    np.savetxt('data/Y_in', sess.run(fc1, feed_dict={x: X_train, y_: y_train}), delimiter=',')
    np.savetxt('data/Y_out', sess.run(fc2, feed_dict={x: X_train, y_: y_train}), delimiter=',')
    # X_train = X_train[0:100,:,:,:]
    # X_test = X_test[0:100,:,:,:]
    n1, n2, n3, n4 = X_train.shape
    n5, n6, n7, n8 = X_test.shape

    np.savetxt('data/train_x', X_train.reshape(n1, n2*n3),  delimiter=',')
    np.savetxt('data/test_x', X_test.reshape(n5, n6*n7),  delimiter=',')
    train_labels = y_train
    test_labels = y_test
    # m1, m2, m3 = train_labels.shape
    # m4, m5, m6 = test_labels.shape
    np.savetxt('data/train_labels', train_labels,  delimiter=',')
    np.savetxt('data/test_labels', test_labels,  delimiter=',')
