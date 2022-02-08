from Functions import *
from ModelOperations import *
from sklearn.ensemble import RandomForestClassifier

# loading the training dataset and splitting it into training and validation set
X_train, y_train = create_train_and_test()

# reducing dimesnions to a variance ratio of 95% and fitting it
X_reduced = pca.fit_transform(X_train)

rnd_clf = RandomForestClassifier(verbose=20)        # random forest classifier
rnd_clf.fit(X_reduced,y_train)

evaluate_model(rnd_clf,"PCA_Reduced_Random_Forest_Classification")