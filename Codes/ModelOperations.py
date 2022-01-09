from Functions import *

from sklearn.datasets import fetch_openml
from numpy import savetxt
from numpy import loadtxt
from scipy.ndimage.interpolation import shift
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
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

def shift_image(image, dx, dy):
    image = image.reshape((28, 28))
    shifted_image = shift(image, [dy, dx], cval=0, mode="constant")
    return shifted_image.reshape([-1])

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

    X_train, y_train = X[:60000],y[:60000]

    X_train_augmented = [image for image in X_train]
    y_train_augmented = [label for label in y_train]

    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        for image, label in zip(X_train, y_train):
            X_train_augmented.append(shift_image(image, dx, dy))
            y_train_augmented.append(label)

    X_train = np.array(X_train_augmented)
    y_train = np.array(y_train_augmented)

    print("Saving datasets to local dir...")
    savetxt(os.path.join(DATASETS,'X_train.csv'), X_train, delimiter=',')
    savetxt(os.path.join(DATASETS,'X_test.csv'), X[60000:], delimiter=',')
    savetxt(os.path.join(DATASETS,'y_train.csv'), y_train, delimiter=',')
    savetxt(os.path.join(DATASETS,'y_test.csv'), y[60000:], delimiter=',')
    print("done")
    
    # By default, MNIST is shuffled and arranged into a test( first 60k isntances) and training set (last 10k instances)
    return X_train, y_train

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
# Only for binary -> ROC, Precision vs Recall, P/R vs threshold, AUC for ROC
def evaluate_model(model,name):

    file_object = open('Model_Performances.txt', 'a')
    file_object.write("\nEvaluation for model "+name+"\n")

    # Making predictions
    print("~~~~~~~~~~~~~~~~~~~~~~ Model Evaluation ~~~~~~~~~~~~~~~~~~~")
    print("Loading test set...")
    X_test = loadtxt(os.path.join(DATASETS,'X_test.csv'), delimiter=',')
    y_test = loadtxt(os.path.join(DATASETS,'y_test.csv'), delimiter=',')
    print("Predicting labels using model....")
    y_pred = model.predict(X_test)

    # Confusion Matrix
    print("Evaluating confusion matrix..")
    conf_mat = confusion_matrix(y_test,y_pred)
    file_object.write("\nConfusion matrix -> \n")
    file_object.write(str(np.array(conf_mat)))

    conf_err(conf_mat,name)

    # Precision, Recall, F1
    print("Evaluating Precision, Recall, F1..")
    file_object.write("\nPrecision -> "+str(precision_score(y_test,y_pred,average="macro")))
    file_object.write("\nRecall -> "+str(recall_score(y_test,y_pred,average="macro")))
    file_object.write("\nF1 score -> "+str(f1_score(y_test,y_pred,average="macro")))
    
    print("Report stored in Model_performances.txt")
    file_object.close()