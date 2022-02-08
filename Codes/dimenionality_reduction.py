from cProfile import label
from Functions import *
from ModelOperations import *
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt

# loading the training dataset and splitting it into training and validation set
X_train, y_train = create_train_and_test()

# using tSNE to reduce dimension of dataset down to 2
if not os.path.isfile(os.path.join(DATASETS,'X_train_reduced_2.csv')):
    t_sne = TSNE(n_components=2,n_jobs=-1)
    X_reduced = t_sne.fit_transform(X_train)
    print(X_reduced.shape)
    savetxt(os.path.join(DATASETS,'X_train_reduced_2.csv'), X_reduced, delimiter=',')
else:
    X_reduced = loadtxt(os.path.join(DATASETS,'X_train_reduced_2.csv'), delimiter=',')
    
plt.scatter(X_reduced[:,0],X_reduced[:,1],c=y_train,cmap="jet")
plt.axis("square")
plt.colorbar()
plt.show()