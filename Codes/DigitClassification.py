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
plot_figure(X_train,y_train,356)