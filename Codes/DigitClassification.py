from Functions import *
from LoadModel import *

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

X_train, X_test, y_train, y_test = create_train_and_test()

param_grid = {
    'weights':['uniform','distance'],
    'n_neighbors':[1,5,10]
}

