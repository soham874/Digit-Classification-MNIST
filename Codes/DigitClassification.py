from sklearn.datasets import fetch_openml

import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# Generic Paths
MODEL_PATH = os.path.join("Models")
IMAGE_PATH = os.path.join("Images")

# Fetching the MNIST Dataset and loading it
mnist  = fetch_openml('mnist_784',version=1,as_frame = False)
X,y = mnist["data"], mnist["target"]

# Verifying Size of dataset
print(X.shape)
print(y.shape)

# function to plot a random digit from the dataset and show its label
def plot_figure(X,y,some_digit):
    some_digit_image = X[some_digit].reshape(28, 28)
    print(y[some_digit])

    plt.imshow(some_digit_image, cmap = mpl.cm.binary, interpolation="nearest")
    plt.axis("off")
    # plt.savefig(os.path.join(IMAGE_PATH,"random_image.png"))
    plt.show()

# Verifying the plot_figure function
plot_figure(X,y,356)

y = y.astype(np.uint8)  # Converting the string type labels into integer type

# By default, MNIST is shuffled and arranged into a test( first 60k isntances) and training set (last 10k instances)
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]