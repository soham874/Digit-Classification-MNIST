from Functions import *
from ModelOperations import *

from sklearn.model_selection import train_test_split

# loading the training dataset and splitting it into training and validation set
X_train, y_train = create_train_and_test()
X_train, x_val, y_train, y_val = train_test_split(X_train,y_train,test_size=0.2)
