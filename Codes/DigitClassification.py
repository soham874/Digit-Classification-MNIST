from Functions import *
from ModelOperations import *

from sklearn.neighbors import KNeighborsClassifier

X_train, y_train = create_train_and_test()

print("Total size of features and labels in training set")
# Verifying Size of dataset
print(X_train.shape)
print(y_train.shape)

# Loading the best model
best_model = load_best_parameters(X_train,y_train,"BestKNClassifier.pkl")
print("Model found with parameters ->")
print(best_model)

# Making predictions and evaluating the model, saving the paramters
print("Fitting model....")
best_model.fit(X_train,y_train)
evaluate_model(best_model,"KNClassifier_augmented_neighbour_4_weight_distance")