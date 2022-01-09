from Functions import *

from sklearn.datasets import fetch_openml
from numpy import savetxt
from numpy import loadtxt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

import os
import joblib
import numpy as np

# Generic Paths
MODEL_PATH = os.path.join("Models")
DATASETS = os.path.join("Datasets")

if not os.path.isdir(MODEL_PATH):
    os.makedirs(MODEL_PATH)
if not os.path.isdir(DATASETS):
    os.makedirs(DATASETS)  

# Function to create the testing and training dataset
def create_train_and_test():

    if os.path.isfile(os.path.join(DATASETS,'X_train.csv')):
        print("Datasets found. Loading...")
        return loadtxt(os.path.join(DATASETS,'X_train.csv'), delimiter=','),loadtxt(os.path.join(DATASETS,'y_train.csv'), delimiter=',')

    print("Datasets not found. Creating new sets...")

    # Fetching the MNIST Dataset and loading it
    mnist  = fetch_openml('mnist_784',version=1,as_frame = False)
    X,y = mnist["data"], mnist["target"]
    y = y.astype(np.uint8)  # Converting the string type labels into integer type

    print("Saving datasets to local dir...")
    savetxt(os.path.join(DATASETS,'X_train.csv'), X[:60000], delimiter=',')
    savetxt(os.path.join(DATASETS,'X_test.csv'), X[60000:], delimiter=',')
    savetxt(os.path.join(DATASETS,'y_train.csv'), y[:60000], delimiter=',')
    savetxt(os.path.join(DATASETS,'y_test.csv'), y[60000:], delimiter=',')
    print("done")
    
    # By default, MNIST is shuffled and arranged into a test( first 60k isntances) and training set (last 10k instances)
    return X[:60000], y[:60000]

# Find the best parameters
def load_best_parameters(X,y,modelname):

    if os.path.isfile(os.path.join(MODEL_PATH,modelname)):
        return joblib.load(os.path.join(MODEL_PATH,modelname))

    param_grid = {
        'weights':['uniform','distance'],
        'n_neighbors':[3,4,5]
    }

    knn_clf = KNeighborsClassifier()
    grid_seach_result = GridSearchCV(knn_clf , param_grid , cv=5, verbose=20)
    grid_seach_result.fit(X, y)

    print(grid_seach_result.best_params_)
    print(grid_seach_result.best_score_)
    joblib.dump(grid_seach_result.best_estimator_,os.path.join(MODEL_PATH,modelname))
    return grid_seach_result.best_estimator_

# Evaluate Confusion Matrix, Precision, Recall, F1 for model
# Plot ROC, Precision vs Recall, P/R vs threshold, AUC for ROC
def evaluate_model(model,X,y,name):
    # Making predictions
    y_pred = model.predict(X)
    # Confusion Matrix
    conf_mat = confusion_matrix(y,y_pred)
    print("~~~~~~~~~ Model Evaluation ~~~~~~~~~~~~~~~~")
    print("Confusion matrix -> ")
    print(conf_mat)

    conf_err(conf_mat,name)