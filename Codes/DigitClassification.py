from Functions import *
from ModelOperations import *

X_train, y_train = create_train_and_test()

print("Total size of features and labels in training set")
# Verifying Size of dataset
print(X_train.shape)
print(y_train.shape)

# Loading the best model
best_model = load_best_parameters(X_train,y_train,"BestSVC.pkl")
print("Model found with parameters ->")
print(best_model)

# Making predictions and evaluating the model, saving the paramters
evaluate_model(best_model,"SVM_Classifer")