from Functions import *
from sklearn.datasets import fetch_openml

import os
import numpy as np

# Generic Paths
MODEL_PATH = os.path.join("Models")
IMAGE_PATH = os.path.join("Images")

def create_train_and_test():
    
    # Fetching the MNIST Dataset and loading it
    mnist  = fetch_openml('mnist_784',version=1,as_frame = False)
    X,y = mnist["data"], mnist["target"]

    # Verifying Size of dataset
    print(X.shape)
    print(y.shape)

    # Verifying the plot_figure function
    plot_figure(X,y,356)

    y = y.astype(np.uint8)  # Converting the string type labels into integer type

    # By default, MNIST is shuffled and arranged into a test( first 60k isntances) and training set (last 10k instances)
    return X[:60000], X[60000:], y[:60000], y[60000:]