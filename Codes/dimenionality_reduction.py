from Functions import *
from ModelOperations import *
# from sklearn.manifold import TSNE
# from sklearn.decomposition import PCA
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.preprocessing import MinMaxScaler
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

import matplotlib.pyplot as plt

# function to plot the reduced dataset
def plot_digits(X, y, min_distance=0.5, images=None, figsize=(13, 10)):
    # Let's scale the input features so that they range from 0 to 1
    X_normalized = MinMaxScaler().fit_transform(X)
    # Now we create the list of coordinates of the digits plotted so far.
    # We pretend that one is already plotted far away at the start, to
    # avoid `if` statements in the loop below
    neighbors = np.array([[10., 10.]])
    # The rest should be self-explanatory
    plt.figure(figsize=figsize)
    cmap = mpl.cm.get_cmap("jet")
    digits = np.unique(y)
    for digit in digits:
        plt.scatter(X_normalized[y == digit, 0], X_normalized[y == digit, 1], c=[cmap(digit / 9)])
    plt.axis("off")
    ax = plt.gcf().gca()  # get current axes in current figure
    for index, image_coord in enumerate(X_normalized):
        closest_distance = np.linalg.norm(neighbors - image_coord, axis=1).min()
        if closest_distance > min_distance:
            neighbors = np.r_[neighbors, [image_coord]]
            if images is None:
                plt.text(image_coord[0], image_coord[1], str(int(y[index])),
                         color=cmap(y[index] / 9), fontdict={"weight": "bold", "size": 16})
            else:
                image = images[index].reshape(28, 28)
                imagebox = AnnotationBbox(OffsetImage(image, cmap="binary"), image_coord)
                ax.add_artist(imagebox)
    plt.show()

# loading the training dataset and splitting it into training and validation set
X_train, y_train = create_train_and_test()

# using tSNE to reduce dimension of dataset down to 2
if not os.path.isfile(os.path.join(DATASETS,'X_train_reduced_2.csv')):
    # dr_model = TSNE(n_components=2,n_jobs=-1)
    # dr_model = PCA(n_components=2)
    pca = PCA(n_components=0.95)
    X_train = pca.fit_transform(X_train)
    dr_model = LocallyLinearEmbedding(n_components=2,n_jobs=-1)
    X_reduced = dr_model.fit_transform(X_train)
    print(X_reduced.shape)
    # savetxt(os.path.join(DATASETS,'X_train_reduced_2.csv'), X_reduced, delimiter=',')
else:
    X_reduced = loadtxt(os.path.join(DATASETS,'X_train_reduced_2.csv'), delimiter=',')
    
plot_digits(X_reduced,y_train)