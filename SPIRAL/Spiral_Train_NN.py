## =================================================================
## =================================================================
from __future__ import print_function

#import tensorflow.contrib.eager as tfe
# Set Eager API
#tfe.enable_eager_execution()
import IPython
import os
cwd = os.getcwd()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Import MNIST data
#from tensorflow.examples.tutorials.m import input_data
#mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


## =================================================================
## =================================================================
## Import Spiral Data 
fname = 'spiral.txt'
data_points = np.genfromtxt('new_spiral.txt', usecols=(0, 1))
data_labels = np.genfromtxt('new_spiral.txt', dtype=str, usecols=(2))
colors = data_labels
# One hot encoding of labels
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(data_labels)
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

## =================================================================
## =================================================================
## Create Train and Test Datasets
msk = np.random.rand(len(data_points)) < 0.8
#np.savetxt('Mask.txt', msk, fmt= '%5i', delimiter=',')
#msk = np.loadtxt(open("Mask.txt"), dtype='int32', skiprows=0) # fix the mask for now
#msk = msk > 0.5
train_x = data_points[msk,].astype(np.float32)
train_y = onehot_encoded[msk,].astype(np.float32)
test_x = data_points[~msk,].astype(np.float32)
test_y = onehot_encoded[~msk,].astype(np.float32)
#msk = np.random.choice(200, 180, replace = False) # try larger data
#np.savetxt('Mask.txt', msk, fmt= '%5i', delimiter=',')
#msk = np.loadtxt(open("Mask.txt"), dtype='int32', skiprows=0) # fix the mask for now
##msk = msk > 0.5
#np.savetxt('Mask.txt', msk, fmt= '%5i', delimiter=',')
#train_x = data_points[msk,].astype(np.float32)
#train_y = onehot_encoded[msk,].astype(np.float32)
#col = colors[msk]
# train_x = data_points.astype(np.float32)
# train_y = onehot_encoded.astype(np.float32)

ind = np.random.choice(train_x.shape[0], 1000, replace='False')
train_x = train_x[ind,:]
train_y = train_y[ind,:]


# 
# IPython.embed()
## =================================================================
## =================================================================
## Initialize Network Parameters
learning_rate = 0.001
n, _ = train_x.shape
batch_size = 50
num_steps = int(np.floor(n/batch_size))
display_step = 1

# Network Parameters
n_hidden_1 = 20 # 1st layer number of neurons
n_hidden_2 = 20 # 2nd layer number of neurotest_t[45:44]ns
#n_hidden_3 = 80
num_input = 2 # Spiral data input 2-D points
num_classes = 2 # Total classes (red or black)

# tf Graph input
X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder("float", [None, num_classes])
## =================================================================
## =================================================================
## Initilaize Layer Weights
# Store layers weight & bias
weights = {
	'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
	'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
	#'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
	'out': tf.Variable(tf.random_normal([n_hidden_2, num_classes]))
}
biases = {
	'b1': tf.Variable(tf.random_normal([n_hidden_1])),
	'b2': tf.Variable(tf.random_normal([n_hidden_2])),
	#'b3': tf.Variable(tf.random_normal([n_hidden_3])),
	'out': tf.Variable(tf.random_normal([num_classes]))
}
## =================================================================
## =================================================================
## Create model
def neural_net(x):
	# Hidden fully connected layer with 200 neurons
	layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
	# Hidden fully connected layer with 200 neurons
	layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))
	# Output fully connected layer with a neuron for each class
	#layer_3 = tf.nn.relu(tf.add(tf.matmul(layer_2, weights['h3']), biases['b3']))
	# Output fully connected layer with a neuron for each class
	out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
	return out_layer

def neural_net_intermediate(x):
	# Hidden fully connected layer with 200 neurons
	layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
	# Hidden fully connected layer with 200 neurons
	layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))
	# Output fully connected layer with a neuron for each class
	#layer_3 = tf.nn.relu(tf.add(tf.matmul(layer_2, weights['h3']), biases['b3']))
	out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
	return layer_1, layer_2#tf.matmul(layer_1, weights['h2'])
## =================================================================
## =================================================================
## Construct model
logits = neural_net(X)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
	logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model (with test logits, for dropout to be disabled)
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))# cast to new data type

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()
## =================================================================
## =================================================================
## Start training
sess=tf.Session()
# Run the initializer
sess.run(init)
for kk in range(1,1250):
	for step in range(1, num_steps+1):## check num_steps batch_sixe !!!!!!!
		batch_x, batch_y = train_x[(batch_size)*(step-1)+1:batch_size*(step),],\
		train_y[(batch_size)*(step-1)+1:batch_size*(step),]  #mnist.train.next_batch(batch_size)
		# Run optimization op (backprop)
		sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
##        if step % display_step == 0 or step == 1:
##             Calculate batch loss and accuracy
##            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
##                                                                    Y: batch_y})
##            print("Step " + str(step) + ", Minibatch Loss= " + \
##                      "{:.4f}".format(loss) + ", Training Accuracy= " + \
##                      "{:.3f}".format(acc))
	if kk % 1 == 0:
		print('     ***     ')
		loss = sess.run(loss_op, feed_dict={X: train_x, Y: train_y})
		print("Epoch " + str(kk) + ", Training Loss= " + "{:.4f}".format(loss))
		acc_test = sess.run(correct_pred, feed_dict={X: test_x, Y: test_y})
		print("Test Accuracy=" + str(np.sum(acc_test)/len(acc_test)))
		acc_train = sess.run(correct_pred, feed_dict={X: train_x, Y: train_y})
		print("Train Accuracy=" + str(np.sum(acc_train)/len(acc_train)))

# Calculate accuracy for MNIST test images
#print("Testing Accuracy:", \
#sess.run(accuracy, feed_dict={X: test_x, Y: test_y})

## =================================================================
## =================================================================
## Test set
#acc = sess.run(correct_pred, feed_dict={X: test_x, Y: test_y})
#prec = np.sum(acc)/len(acc)
#print('Precicion on test set is: {}'.format(prec))


# watch out whether before  after activation function


# Do the prediction Manually
# Have matrix with LDR weights 
def neural_net_LDR(x, weights_LDR):
	# Hidden fully connected layer with 200 neurons
	layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
	# Hidden fully connected layer with 200 neurons
	layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights_LDR), biases['b2']))
	# Output fully connected layer with a neuron for each class
	out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
	return out_layer


## =================================================================
## =================================================================
# Compute error in the same train set for LDR matrices

# nn_out = sess.run(neural_net(test_x))
# correct_pred = tf.equal(tf.argmax(nn_out,1), tf.argmax(test_y, 1))
# accuracy = sess.run(tf.reduce_mean(tf.cast(correct_pred, tf.float32)))
# print(accuracy)

# Read LDR matrix
#W_LDR = np.loadtxt(open("UMatrix0001.txt", "rb"), dtype='float32', delimiter=",", skiprows=0)
#W_LDR = np.random.uniform(-2,2,(80, 80)).astype(np.float32)
#nn_out_LDR = sess.run(neural_net_LDR(train_x, W_LDR))
#correct_pred_LDR = tf.equal(tf.argmax(nn_out_LDR, 1), tf.argmax(train_y, 1))
#accuracy_LDR = sess.run(tf.reduce_mean(tf.cast(correct_pred_LDR, tf.float32)))
#print('accuracy LDR={}'.format(accuracy_LDR))





## =================================================================
## =================================================================
## Plots
## Plot Spiral

##plt.scatter(data_points[:,0], data_points[:,1], c=colors)
##axes = plt.gca()
##axes.set_xlim([-15,15])
##axes.set_ylim([-15,15])
##plt.show()
## =================================================================
## =================================================================
# Plot decision boundary


## =================================================================
## =================================================================

def neural_net_forget(x):
	# Hidden fully connected layer with 200 neurons
	layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
	# Hidden fully connected layer with 200 neurons
	layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))
	#layer_3 = tf.nn.relu(tf.add(tf.matmul(layer_2, weights['h3']), biases['b3']))
	# Output fully connected layer with a neuron for each class
	out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
	return biases['b2'], weights['h2'], layer_1, layer_2

Bias, Weight, layer_1, layer_2 = sess.run(neural_net_forget(train_x))
np.savetxt('data/Weight.txt', Weight, delimiter=',')
np.savetxt('data/Bias.txt', Bias, delimiter=',')
np.savetxt('data/layer_in.txt', layer_1, delimiter=',')
np.savetxt('data/layer_out.txt', layer_2, delimiter=',')

np.savetxt('data/first_bias.txt', sess.run(biases['b1']), delimiter=',')
np.savetxt('data/last_bias.txt', sess.run(biases['out']), delimiter=',')
np.savetxt('data/first_weight.txt', sess.run(weights['h1']), delimiter=',')
np.savetxt('data/last_weight.txt', sess.run(weights['out']), delimiter=',')

#np.savetxt('lastbias.txt', lastbias, delimiter=',')

np.savetxt('data/train_y.txt', train_y, delimiter=',')
np.savetxt('data/train_x.txt', train_x, delimiter=',')
np.savetxt('data/test_y.txt', test_y, delimiter=',')
np.savetxt('data/test_x.txt', test_x, delimiter=',')



