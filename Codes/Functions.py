import matplotlib as mpl
import matplotlib.pyplot as plt

import os
import numpy as np

IMAGE_PATH = os.path.join("Images")

if not os.path.isdir(IMAGE_PATH):
    os.makedirs(IMAGE_PATH)

# function to plot a random digit from the dataset and show its label
def plot_figure(X,y,some_digit,name=""):
    some_digit_image = X[some_digit].reshape(28, 28)
    print(y[some_digit])

    plt.imshow(some_digit_image, cmap = mpl.cm.binary, interpolation="nearest")
    plt.axis("off")

    if name != "":
        plt.savefig(os.path.join(IMAGE_PATH,name))

    plt.show()

# function to save confusion and its error matrix
def conf_err(conf_mat,name):

    if not os.path.isdir(os.path.join(IMAGE_PATH,name)):
        os.makedirs(os.path.join(IMAGE_PATH,name))

    # confusion matix
    plt.matshow(conf_mat, cmap=plt.cm.gray)
    plt.savefig(os.path.join(IMAGE_PATH,name,"Confusion_matrix.png"))

    row_sums = conf_mat.sum(axis=1, keepdims=True)
    norm_conf_mat = conf_mat / row_sums

    np.fill_diagonal(norm_conf_mat, 0)
    plt.matshow(norm_conf_mat, cmap=plt.cm.gray)
    plt.savefig(os.path.join(IMAGE_PATH,name,"Error_matrix.png"))
