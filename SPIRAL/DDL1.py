import cvxpy as cvx
import numpy as np
from cvxpy import *
import IPython
import numpy.matlib

# function used for output file name
def decimal_str(x: float, decimals: int = 10) -> str:
    return format(x, f".{decimals}f").lstrip().rstrip('0')

# load data
W = np.loadtxt(open("data/Weight.txt", "rb"), dtype='float32', delimiter=",", skiprows=0)
b = np.loadtxt(open("data/Bias.txt", "rb"), dtype='float32', delimiter=",", skiprows=0)
X = np.loadtxt(open("data/layer_in.txt", "rb"), dtype='float32', delimiter=",", skiprows=0)
Y = np.loadtxt(open("data/layer_out.txt", "rb"), dtype='float32', delimiter=",", skiprows=0)


m, n = X.shape
# number of data points to solve the optimization
N = 500 
np.random.seed(8)
ind = np.random.choice(m, N, replace=False)


# reshape bias for calculations
b_temp = np.expand_dims(b, axis=1)
bb = np.matlib.repmat(b_temp.T, N, 1) 


# Select sub-sample of data for approximation
X_small = X[ind, :]
Y_small = Y[ind, :]


# Mask matrices to select the positive and negative elements of output
Mpos = np.zeros((N, n))
Mneg = np.zeros((N, n))
for i in range(N):
	for j in range(n):
		if Y_small[i, j] > 0:
			Mpos[i, j] = 1
		else:
			Mneg[i, j] = 1

# file to save results
myfile = open('data/Spiral_Constr_L1_N_' + str(N) + '_notes_python.txt', 'w')

# RHS inequality constraint scaling
const = np.linalg.norm(X_small,'fro')


for epsilon in [ 0.0001, 0.001, 0.01, 0.1, 1, 10, 100]:
	
	# Optimization
	U = Variable((n, n))
	obj = cvx.Minimize(cvx.mixed_norm(U, 1, 1))
	constraints = [cvx.norm(cvx.multiply(X_small @ U + bb - Y_small, Mpos), 'fro') <= epsilon * const, 
	cvx.multiply(X_small @ U + bb - Y_small, Mneg) <= 0]
	prob = cvx.Problem(obj, constraints)
	print("Optimal value", prob.solve(solver=cvx.GUROBI, verbose=True, max_iters = 30000))#solver=cvx.GUROBI,


	# Compute sparsity
	nnz = np.count_nonzero(U.value > 0.0001)
	print("Number os nnz is {}".format(nnz))

	# Save data
	filename = 'data/Spiral_Constr_L1_N_' + str(N) + '_Eps_' + str(decimal_str(epsilon)).replace('.', '') + '.txt'
	np.savetxt(filename, U.value,  delimiter=',')
	myfile.write('{},{}\n'.format(np.format_float_positional(np.float16(epsilon)), nnz))


myfile.close()