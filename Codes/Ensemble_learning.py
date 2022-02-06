from Functions import *
from ModelOperations import *

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, StackingClassifier

# loading the training dataset and splitting it into training and validation set
X_train, y_train = create_train_and_test()
X_train, x_val, y_train, y_val = train_test_split(X_train,y_train,test_size=0.2)

extra_clf = ExtraTreesClassifier()        # extra forest model
rnd_clf = RandomForestClassifier()        # random forest classifier
svm_clf = SVC(probability=True)          # Support vector classifier

# specifying all models on which the voting should be done
# voting_clf = VotingClassifier( 
#    estimators=[('lr', extra_clf), ('rf', rnd_clf), ('svc', svm_clf)], 
#    voting='soft')
# voting_clf.fit(X_train, y_train)

# evaluate_model(voting_clf,"Soft_Voting_Classifier")

# Outputting the accuracies of each model seperately
# for clf in (extra_clf, rnd_clf, svm_clf, voting_clf):
#    clf.fit(X_train, y_train)
#    y_pred = clf.predict(x_val)
#    print(clf.__class__.__name__," -> ",accuracy_score(y_val, y_pred))

stack_clf = StackingClassifier( 
    estimators=[('lr', extra_clf), ('rf', rnd_clf), ('svc', svm_clf)], # all classifiers in the ensemble
    final_estimator = SVC(),       # blender model, final classifer
    n_jobs = -1,
    cv = 5,                        # cross validation folds         
    verbose = 20
    )
stack_clf.fit(X_train, y_train)

evaluate_model(stack_clf,"Blender_Ensemble_Classification")