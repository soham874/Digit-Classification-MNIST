from Functions import *
from ModelOperations import *

X_train, y_train = create_train_and_test()

print("Total size of features and labels in training set")
# Verifying Size of dataset
print(X_train.shape)
print(y_train.shape)

# print("Plotting random figure from dataset")
# Verifying the plot_figure function
# plot_figure(X_train,y_train,356)

# Loading the best model
best_model = load_best_parameters(X_train,y_train,"BestKNClassifier.pkl")
print("Model found with parameters ->")
print(best_model)