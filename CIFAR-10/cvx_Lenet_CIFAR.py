import cvxpy as cvx
import numpy as np
from numpy import linalg as LA
from cvxpy import *
import keras
from keras import optimizers
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from keras.callbacks import LearningRateScheduler, TensorBoard
from keras.preprocessing.image import ImageDataGenerator
import IPython 
from keras.models import load_model
from keras import backend as K
from numpy import matlib 

def decimal_str(x: float, decimals: int = 10) -> str:
    return format(x, f".{decimals}f").lstrip().rstrip('0')


# Just Low Rank
def low_rank_k(u,s,vh,num):
# rank k approx
    u = u[:,:num]
    vh = vh[:num,:]
    s = s[:num]
    s = np.diag(s)
    my_low_rank = np.dot(np.dot(u,s),vh)
    return my_low_rank




batch_size    = 128
epochs        = 200
iterations    = 391
num_classes   = 10
mean          = [125.307, 122.95, 113.865]
std           = [62.9932, 62.0887, 66.7048]
 # load CIFAR-10 data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# data preprocessing  [raw - mean / std]
for i in range(3):
    x_train[:,:,:,i] = (x_train[:,:,:,i] - mean[i]) / std[i]
    x_test[:,:,:,i] = (x_test[:,:,:,i] - mean[i]) / std[i]

m, _, _, _ = x_train.shape

model = load_model('lenet_dp_da.h5')
weights = model.get_weights()
W = weights[6]
b = weights[7]
# W2 = weights[6]
# B2 = weights[7]

np.random.seed(8)
N = 10000
ind = np.random.choice(m, N, replace=False)
X_small_sample = x_train[ind,:,:,:]
Y_small_sample = y_train[ind,:]

#######
input1 = model.input               # input placeholder
IPython.embed()
output1 = [layer.output for layer in model.layers[-3:-1]]# all layer outputs ## -4:-2 for 1st layer, -3:-1 for 2nd layer
fun = K.function([#, K.learning_phase()], output1)# evaluation function

layer_outputs = fun([X_small_sample])
X_small = layer_outputs[0]
Y_small = layer_outputs[1]

#IPython.embed()
#print layer_outputs// printing the outputs of layers
########
#IPython.embed()
n1, n2 = W.shape
U = np.zeros((n1, n2))

_, n = Y_small.shape

for N in [10000]:
	# ind = np.random.choice(m, N, replace=False)
	# # train_x_small = train_x[ind,:,:,:]
	# # train_labels_small = train_labels[ind,:]
	# X_small = X[ind, :]
	# Y_small = Y[ind, :]
	b_temp = np.expand_dims(b, axis=1)
	bb = np.matlib.repmat(b_temp.T, N, 1) # manipulating bias dimension for addition


	# save ranks of the approximation
	myfile = open('data/CIFAR_layer2_Constr_low_rank_N_' + str(N) + '_notes_python.txt', 'w')
	const = np.linalg.norm(X_small,'fro')
	cnt = 0
	for epsilon in [0.01,0.02,0.04,0.06,0.1,0.15,0.2,0.25,0.3,0.4]:#0.01,0.02,0.04,0.06,0.1,0.15,0.2,0.25,0.3,0.4
		# Select sub-sample of data for approximation # for first layer try also 0.5,0.6,0.63,0.67, 0.7
		
		print('*************************** Epsilon is {}*************'.format(epsilon))

		# Mask matrices
		Mpos = np.zeros((N, n))
		Mneg = np.zeros((N, n))

		for i in range(N):
			for j in range(n):
				if Y_small[i, j] > 0:
					Mpos[i, j] = 1
				else:
					Mneg[i, j] = 1

		# Optimization
		U = Variable((n1, n2))
		# if cnt == 0:
		# 	u, s, v = np.linalg.svd(W)
		# 	U.value = low_rank_k(u, s , v, 35)
		# else:
		# 	U.value = opt_var


		obj = cvx.Minimize(cvx.norm(U, 'nuc'))
		constraints = [cvx.norm(cvx.multiply(X_small @ U + bb - Y_small, Mpos), 'fro') <= epsilon * const, 
		cvx.multiply(X_small @ U + bb - Y_small, Mneg) <= 0]
		prob = cvx.Problem(obj, constraints)
		print("Optimal value", prob.solve(solver=cvx.SCS, verbose=True, max_iters = 30000))#scale=100

		#opt_var = U.value
		# Compute rank
		[Utemp,S,V] = np.linalg.svd(U.value )
		rank = np.count_nonzero(S > 0.001)
		print("Rank is {}".format(rank))
		#IPython.embed()
		# Save data
		filename = 'data/CIFAR_layer2_Constr_Low_Rank_N_' + str(N) + '_Eps_' + str(decimal_str(epsilon)).replace('.', '') + '.txt'
		np.savetxt(filename, U.value,  delimiter=',')
		myfile.write('{},{}\n'.format(np.format_float_positional(np.float16(epsilon)), rank))
		cnt += 1

	myfile.close()