import matplotlib as mpl
import matplotlib.pyplot as plt
import os

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
    return plt