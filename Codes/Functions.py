import matplotlib as mpl
import matplotlib.pyplot as plt

# function to plot a random digit from the dataset and show its label
def plot_figure(X,y,some_digit):
    some_digit_image = X[some_digit].reshape(28, 28)
    print(y[some_digit])

    plt.imshow(some_digit_image, cmap = mpl.cm.binary, interpolation="nearest")
    plt.axis("off")
    # plt.savefig(os.path.join(IMAGE_PATH,"random_image.png"))
    plt.show()
    return plt