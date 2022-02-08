from Functions import *
from ModelOperations import *
from sklearn.ensemble import RandomForestClassifier

# loading the training dataset and splitting it into training and validation set
X_train, y_train = create_train_and_test()


rnd_clf = RandomForestClassifier(verbose=20)        # random forest classifier
rnd_clf.fit(X_train,y_train)

evaluate_model(rnd_clf,"Random_Forest_Classification")