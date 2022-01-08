from sklearn.datasets import fetch_openml
import matplotlib as mpl
import matplotlib.pyplot as plt

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
    plt.show()

# Verifying the plot_figure function
plot_figure(X,y,356)