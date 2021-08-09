## =================================================================
## =================================================================
# Plot decision boundary
import numpy as np
import matplotlib.pyplot as plt

fname = 'spiral.txt'
data_points = np.genfromtxt('new_spiral.txt', usecols=(0, 1))
data_labels = np.genfromtxt('new_spiral.txt', dtype=str, usecols=(2))
colors = np.zeros(len(data_labels))
for i in range(len(data_labels)):
    if(data_labels[i] == '1'):
        colors[i] = 1
layer_1 = np.loadtxt(open("layer_1_out.txt", "rb"), dtype='float32', delimiter=",", skiprows=0)
W1 = np.loadtxt(open("Weight_eps80_u.txt", "rb"), dtype='float32', delimiter=",", skiprows=0)
#W2 = np.loadtxt(open("Weight_2_5_plot.txt", "rb"), dtype='float32', delimiter=",", skiprows=0)
B1 = np.loadtxt(open("firstbias.txt", "rb"), dtype='float32', delimiter=",", skiprows=0)
#B2 = np.loadtxt(open("secondbias.txt", "rb"), dtype='float32', delimiter=",", skiprows=0)
W3 = np.loadtxt(open("lastweight.txt", "rb"), dtype='float32', delimiter=",", skiprows=0)
B3 = np.loadtxt(open("lastbias.txt", "rb"), dtype='float32', delimiter=",", skiprows=0)
X = np.loadtxt(open("train_x.txt", "rb"), dtype='float32', delimiter=",", skiprows=0)
W0 = np.loadtxt(open("W0.txt", "rb"), dtype='float32', delimiter=",", skiprows=0)
B0 = np.loadtxt(open("B0.txt", "rb"), dtype='float32', delimiter=",", skiprows=0)
#msk = np.random.rand(len(data_points)) < 0.1
#X = X[msk,]
#colors = colors[msk,]
def neural_net_plot(X,W0, W1, W3, B0, B1, B3):
    # Hidden fully connected layer with 200 neurons
    layer_1 = np.maximum(np.matmul(X, W0)+ B0, 0)
    # Hidden fully connected layer with 200 neurons
    layer_2 = np.maximum(np.matmul(layer_1, W1) + B1,0)
    # Output fully connected layer with a neuron for each class
    #layer_3 = np.maximum(np.matmul(layer_2, W2) + B2,0)
    # Output fully connected layer with a neuron for each class
    logits = np.matmul(layer_2, W3) + B3
    out_layer = np.argmax(logits, 1)
    return out_layer

#def pred_func(X, layer_1, W1, W2, W3, B1, B2, B3):
#    logits = neural_net_plot(X, layer_1, W1, W2, W3, B1, B2, B3)
#    lab = sess.run(tf.argmax(logits, 1))
#    return(lab)

# Helper function to plot a decision boundary.
# If you don't fully understand this function don't worry, it just generates the contour plot below.
def plot_decision_boundary(neural_net_plot, X, W0, W1, W3, B0, B1, B3, col):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    x_min = np.float32(x_min)
    x_max = np.float32(x_max)
    y_min = np.float32(y_min)
    y_max = np.float32(y_max)
    h = 0.1
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    xx = np.float32(xx)
    yy = np.float32(yy)
    Z = neural_net_plot(np.c_[xx.ravel(), yy.ravel()], W0, W1,  W3, B0, B1,  B3)
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], s=1.5, c=col)
    plt.show()


plot_decision_boundary(neural_net_plot, X, W0, W1,  W3, B0, B1, B3, colors)



W1 = np.loadtxt(open("Weight1.txt", "rb"), dtype='float32', delimiter=",", skiprows=0)

W2 = np.loadtxt(open("Weight_eps80_u.txt", "rb"), dtype='float32', delimiter=",", skiprows=0)

W3 = np.loadtxt(open("Weight_eps80_ldr.txt", "rb"), dtype='float32', delimiter=",", skiprows=0)


plt.subplot(1, 3, 1)
plt.imshow(W1)
plt.subplot(1, 3, 2)
plt.imshow(W2)
plt.subplot(1, 3, 3)
plt.imshow(W3)

