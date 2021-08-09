import cvxpy as cvx
import numpy
# Problem data.
import numpy as np
from numpy import linalg as LA
from cvxpy import *
import IPython
import numpy.matlib
import sdpt3glue

W = np.loadtxt(open("data/Weight.txt", "rb"), dtype='float32', delimiter=",", skiprows=0)
b = np.loadtxt(open("data/Bias.txt", "rb"), dtype='float32', delimiter=",", skiprows=0)
X = np.loadtxt(open("data/layer_in.txt", "rb"), dtype='float32', delimiter=",", skiprows=0)
Y = np.loadtxt(open("data/layer_out.txt", "rb"), dtype='float32', delimiter=",", skiprows=0)

np.random.seed(3)


m, n = X.shape
Z1 = np.zeros((n, n))
Zm1 = np.zeros((n, n))
for i in range(n-1):
    Z1[i+1, i] = 1
    Zm1[i+1, i] = 1
Z1[0, n-1] = 1
Zm1[0, n-1] = -1
# Initialize matrices
U = np.zeros((n, n))
#m = 8
#X = X[0:8,0:8]
#Y = Y[0:8,0:8]
# select subset of data for approximation
N = 100  # number of points to solve the optimization
myfile = open('data/SDPT3notes_python_N_' + str(N) + '.txt', 'w')

b_temp = np.expand_dims(b, axis=1)
bb = np.matlib.repmat(b_temp.T, N, 1) # manipulating bias dimension for addition
np.random.seed(3)
ind = np.random.choice(m, N, replace=False)
#ind = np.loadtxt(open("indices_50.txt", "rb"), dtype='float32', delimiter=",").astype(int)
for epsilon in [0.01]:# Frobenious norm RHS
	# Select sub-sample of data for approximation
	X_small = X[ind, :]
	Y_small = Y[ind, :]
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
	U = Variable((n, n))
	obj = cvx.Minimize(cvx.norm(U, 'nuc'))
	constraints = [cvx.norm(cvx.multiply(X_small @ U + bb - Y_small, Mpos), 'fro') <= epsilon, cvx.multiply(X_small @ U + bb - Y_small, Mneg) <= 0]
	prob = cvx.Problem(obj, constraints)
	#print("Optimal value", prob.solve(solver=cvx.SDPT3,verbose=True))
	matfile_target = os.path.join(folder, 'matfile.mat')  # Where to save the .mat file to
	output_target = os.path.join(folder, 'output.txt')    # Where to save the output log
	result = sdpt3glue.sdpt3_solve_problem(prob, sdpt3glue.MATLAB, matfile_target,
                                       output_target=output_target)
	# Compute rank
	[Utemp,S,V] = np.linalg.svd(U.value)
	rank = np.count_nonzero(S > 0.001)
	print("Rank is {}".format(rank))

	# Save data
	filename = 'data/SDPT3Low_Rank_N_' + str(N) + '_Eps_' + str(epsilon).replace('.', '') + '.txt'
	np.savetxt(filename, U.value,  delimiter=',')
	myfile.write("Epsilon is {}, rank is {}, N = {}\n".format(epsilon, rank, N))


myfile.close()