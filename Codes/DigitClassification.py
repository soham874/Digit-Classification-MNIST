from Functions import *
from LoadModel import *

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

X_train, X_test, y_train, y_test = create_train_and_test()

print("Total size of features and labels set")
# Verifying Size of dataset
print(X_train.shape)
print(y_train.shape)

print("Plotting random figure from dataset")
# Verifying the plot_figure function
# plot_figure(X_train,y_train,356)

param_grid = {
    'weights':['uniform','distance'],
    'n_neighbors':[3,4,5]
}

knn_clf = KNeighborsClassifier()
grid_seach_result = GridSearchCV(knn_clf , param_grid , cv=5, verbose=20)

grid_seach_result.fit(X_train, y_train)

grid_seach_result.best_params_
grid_seach_result.best_score_