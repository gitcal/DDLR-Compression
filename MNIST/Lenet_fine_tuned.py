import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import IPython
import IPython
import os
cwd = os.getcwd()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from sklearn.utils import shuffle
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
from sklearn.model_selection import train_test_split
from tensorflow.contrib.layers import flatten
import pickle


def low_rank_k(u, s, vh, num):
	# function for low rank approximation

	u = u[:,:num]
	vh = vh[:num,:]
	s = s[:num]
	s = np.diag(s)
	low_rank1 = np.dot(u, s)
	low_rank2 = vh
	return low_rank1, low_rank2


def decimal_str(x: float, decimals: int = 10) -> str:
	return format(x, f".{decimals}f").lstrip().rstrip('0')


def get_compression_accuracy(x, y, conv1_w, conv1_b, conv2_w ,conv2_b, fc1_w, fc1_b, fc2_w, fc2_b, fc3_w, fc3_b):
	conv1 = tf.nn.conv2d(x,conv1_w, strides = [1,1,1,1], padding = 'VALID') + conv1_b 
	# TODO: Activation.
	conv1 = tf.nn.relu(conv1)
	# Pooling Layer. Input = 28x28x1. Output = 14x14x6.
	pool_1 = tf.nn.max_pool(conv1, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID')
	# TODO: Layer 2: Convolutional. Output = 10x10x16.
	conv2 = tf.nn.conv2d(pool_1, conv2_w, strides = [1,1,1,1], padding = 'VALID') + conv2_b
	# TODO: Activation.
	conv2 = tf.nn.relu(conv2)# TODO: Pooling. Input = 10x10x16. Output = 5x5x16.
	pool_2 = tf.nn.max_pool(conv2, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID')
	# TODO: Flatten. Input = 5x5x16. Output = 400.
	fc1 = flatten(pool_2)
	# TODO: Layer 3: Fully Connected. Input = 400. Output = 120.
	fc1 = tf.matmul(fc1,fc1_w) + fc1_b
	# TODO: Activation.
	fc1 = tf.nn.relu(fc1)
	# TODO: Layer 4: Fully Connected. Input = 120. Output = 84.
	fc2 = tf.matmul(fc1, fc2_w) + fc2_b
	# TODO: Activation.
	fc2 = tf.nn.relu(fc2)
	# TODO: Layer 5: Fully Connected. Input = 84. Output = 10.
	#IPython.embed()
	logits = tf.matmul(tf.cast(fc2, tf.float32), fc3_w) + fc3_b

	correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(y,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))# Evaluate function
	sess = tf.Session()
	return sess.run(accuracy)



def fine_tune(N, eps, epochs, rank1, rank2):

	x = tf.placeholder(tf.float32, shape=[None,32,32,1])
	y_ = tf.placeholder(tf.int32, (None))

	conv1_w = np.loadtxt('data/conv1_w', delimiter=',').reshape([5,5,1,6])
	conv1_b = np.loadtxt('data/conv1_b',  delimiter=',')
	conv2_w = np.loadtxt('data/conv2_w', delimiter=',').reshape([5,5,6,16])
	conv2_b = np.loadtxt('data/conv2_b', delimiter=',')
	fc1_w = np.loadtxt('data/fc1_w', delimiter=',')
	fc1_b = np.loadtxt('data/fc1_b', delimiter=',')
	fc2_w = np.loadtxt('data/fc2_w', delimiter=',')
	fc2_b = np.loadtxt('data/fc2_b', delimiter=',')
	fc3_w = np.loadtxt('data/fc3_w', dtype = np.float32, delimiter=',')
	fc3_b = np.loadtxt('data/fc3_b', dtype = np.float32, delimiter=',')
	Y_in = np.loadtxt('data/Y_in', delimiter=',')
	Y_out = np.loadtxt('data/Y_out', delimiter=',')
	train_x = np.loadtxt('data/train_x', delimiter=',').reshape([33600, 32, 32, 1])
	test_x = np.loadtxt('data/test_x', delimiter=',').reshape([8400, 32, 32, 1])
	train_y = np.loadtxt('data/train_labels', delimiter=',')
	test_y = np.loadtxt('data/test_labels', delimiter=',')


	# tf Graph input
	x = tf.placeholder(tf.float32, shape=[None,32,32,1])
	y_ = tf.placeholder(tf.int32, (None))
	## =================================================================
	## =================================================================
	# laod notes for compression

	file_name_1 = 'Lenet_layer1_Constr_Low_Rank_N_' + str(N) + '_Eps_' + str(decimal_str(eps)).replace('.', '')
	file_name_2 = 'Lenet_2nd_layer_Constr_Low_Rank_N_' + str(N) + '_Eps_' + str(decimal_str(eps)).replace('.', '')
	# load first layer
	weight_lr1 = np.loadtxt('data/'+ file_name_1 + '.txt', delimiter=',')
	weight_lr2 = np.loadtxt('data/'+ file_name_2 + '.txt', delimiter=',')
	
	u, s, vh = np.linalg.svd(weight_lr1, full_matrices = False)
	W_lr_to_tune_1_1, W_lr_to_tune_1_2 = low_rank_k(u, s, vh, rank1)

	u, s, vh = np.linalg.svd(weight_lr2, full_matrices = False)
	W_lr_to_tune_2_1, W_lr_to_tune_2_2 = low_rank_k(u, s, vh, rank2)
	# Initialize the variables (i.e. assign their default value)
	## Initiliaze Layer Weights
	# Store layers weight & bias
	weights = {
		'h1_1': tf.Variable(W_lr_to_tune_1_1, dtype=tf.float32),
		'h1_2': tf.Variable(W_lr_to_tune_1_2, dtype=tf.float32),
		'h2_1': tf.Variable(W_lr_to_tune_2_1, dtype=tf.float32),
		'h2_2': tf.Variable(W_lr_to_tune_2_2, dtype=tf.float32)
	}
	## =================================================================
	## =================================================================
	## Create model
	def neural_net(x, conv1_w, conv1_b, conv2_w ,conv2_b, fc1_w, fc1_b, fc2_w, fc2_b, fc3_w, fc3_b):
		# Network architecture with only middle layer being a variable now
		# layer_1 = tf.nn.relu(tf.add(tf.matmul(x, Win), Bin))
		# layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, tf.matmul(weights['h2_1'], weights['h2_2'])), Btest)) # I am using trained bias without fine tuning it
		# out_layer = tf.matmul(layer_2, Wout) + Bout
		conv1 = tf.nn.conv2d(x,conv1_w, strides = [1,1,1,1], padding = 'VALID') + conv1_b 
		# TODO: Activation.
		conv1 = tf.nn.relu(conv1)
		# Pooling Layer. Input = 28x28x1. Output = 14x14x6.
		pool_1 = tf.nn.max_pool(conv1, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID')
		# TODO: Layer 2: Convolutional. Output = 10x10x16.
		conv2 = tf.nn.conv2d(pool_1, conv2_w, strides = [1,1,1,1], padding = 'VALID') + conv2_b
		# TODO: Activation.
		conv2 = tf.nn.relu(conv2)# TODO: Pooling. Input = 10x10x16. Output = 5x5x16.
		pool_2 = tf.nn.max_pool(conv2, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID')
		# TODO: Flatten. Input = 5x5x16. Output = 400.
		fc1 = flatten(pool_2)
		# TODO: Layer 3: Fully Connected. Input = 400. Output = 120.
		fc1 = tf.matmul(fc1, tf.matmul(weights['h1_1'], weights['h1_2'])) + fc1_b
		# TODO: Activation.
		fc1 = tf.nn.relu(fc1)
		# TODO: Layer 4: Fully Connected. Input = 120. Output = 84.
		fc2 = tf.matmul(fc1, tf.matmul(weights['h2_1'], weights['h2_2'])) + fc2_b
		# TODO: Activation.
		fc2 = tf.nn.relu(fc2)
		logits = tf.matmul(fc2, fc3_w) + fc3_b
		return logits


	
	# learning_rate = 0.001
	## Construct model
	logits = neural_net(x, conv1_w, conv1_b, conv2_w ,conv2_b, fc1_w, fc1_b, fc2_w, fc2_b, fc3_w, fc3_b)


	#Invoke LeNet function by passing features
	#logits,   conv1_w, conv1_b, conv2_w, conv2_b, fc1_w, fc1_b, fc2_w, fc2_b, fc3_w, fc3_b, fc1, fc2 = LeNet_5(x)#Softmax with cost function implementation
	cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels = y_, logits = logits)
	loss_operation = tf.reduce_mean(cross_entropy)
	optimizer = tf.train.AdamOptimizer(learning_rate = 0.001)
	training_operation = optimizer.minimize(loss_operation)


	correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(y_,1))
	accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))# Evaluate function
	
	
	



	#To initialise session and run
	BATCH_SIZE = 128
	np.random.seed(3)
	#N = 200
	m, _, _, _ = train_x.shape
	ind = np.random.choice(m, N, replace=False)
	X_train = train_x[ind,:,:,:]
	y_train = train_y[ind, :]

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		num_examples = len(X_train)
		
		print("Training... with dataset - ", num_examples)
		print()
		for i in range(epochs):
			X_train, y_train = shuffle(X_train, y_train)
			for offset in range(0, num_examples, BATCH_SIZE):
				end = offset + BATCH_SIZE
				batch_x, batch_y = X_train[offset:end], y_train[offset:end]
				sess.run(training_operation, feed_dict={x: batch_x, y_: batch_y})
				


		num_examples = len(test_x)
		total_accuracy = 0

		#sess = tf.get_default_session()
		for offset in range(0, num_examples, BATCH_SIZE):
			batch_x, batch_y = test_x[offset:offset+BATCH_SIZE], test_y[offset:offset+BATCH_SIZE]
			#IPython.embed()
			accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y_: batch_y})
			total_accuracy += (accuracy * len(batch_x))
		 

		acc_test_ft = total_accuracy / num_examples
			#validation_accuracy = evaluate(X_validation, y_validation)
			# print("EPOCH {} ...".format(i+1))
			# train_accuracy = evaluate(X_train, y_train)
			# print("Train Accuracy = {:.3f}".format(train_accuracy))
			# #print()	    
			# test_accuracy = evaluate(test_x, test_y)
			# print("Test Accuracy = {:.3f}".format(test_accuracy))
	# this should generate the same data samples used for constrained low rank
	#np.random.seed(3)
	#N = 200
	# m, _, _, _ = train_x.shape
	# ind = np.random.choice(m, N, replace=False)
	# test_x_N = test_x[ind,]
	# test_y_N = test_y[ind,]
	#IPython.embed()
	
	return acc_test_ft






res_dict = {}
# for epochs in [1, 2, 5]:
# 	print("Epoch {}".format(epochs))
for N in [200, 400, 800]:
	print("N {}".format(N))
	notes_filename_1 = 'Lenet_layer1_Constr_low_rank_N_' + str(N) + '_notes_python'
	notes_1 = np.loadtxt('data/' + notes_filename_1 + '.txt', delimiter=',')
	n_notes_1, _ = notes_1.shape
	epsilon_1 = notes_1[:, 0]
	ranks_1 = notes_1[:, 1]

	notes_filename_2 = 'Lenet_Constr_low_rank_N_' + str(N) + '_notes_python'
	notes_2 = np.loadtxt('data/' + notes_filename_2 + '.txt', delimiter=',')
	n_notes_2, _ = notes_2.shape
	epsilon_2 = notes_2[:, 0]
	ranks_2 = notes_2[:, 1]
	cnt = 0
	for eps in epsilon_1:
		
		# start fine tuning
		acc = fine_tune(N, eps, 0, int(ranks_1[cnt]), int(ranks_2[cnt]))
		cnt += 1
		res_dict[(N,  eps)] = acc


f = open("data/MNIST_FT_dict_no_epochs.pkl","wb")
pickle.dump(res_dict,f)
f.close()

IPython.embed()