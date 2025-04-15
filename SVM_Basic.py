
# import required libraries

import pandas as pd # data loading
from sklearn.model_selection import train_test_split # allows us to split dataset for training/testing
from sklearn.svm import SVC # SVM classifier from scikit-learn
from sklearn.metrics import confusion_matrix # for evaluating model performance
from sklearn.preprocessing import LabelEncoder # for converting categorical labels (like “apple") into numeric values

# read data from csv file
# loads the csv file, must have 3 columns (2 features and 1 label)
data = pd.read_csv('apples_and_oranges.csv')
#print(data)

# splitting data into training and test set
# randomly splits dataset, 20% testing and 80% training
training_set,test_set = train_test_split(data,test_size=0.2,random_state=1)
#print("train:",training_set)
#print("test:",test_set)

# prepare data for applying it to svm
# Selects the first two columns as features (X) and the third column as the label (y).
x_train = training_set.iloc[:,0:2].values  # data
y_train = training_set.iloc[:,2].values  # target
x_test = test_set.iloc[:,0:2].values  # data
y_test = test_set.iloc[:,2].values  # target
#print(x_train,y_train)
#print(x_test,y_test)

# fitting the data (train a model)
# creates and trains an SVM model with RBF kernel (non-linear), C=1 regularization strength, 
# gamma='auto' controls how much influence each training example has
classifier = SVC(kernel='rbf',random_state=1,C=1,gamma='auto')
classifier.fit(x_train,y_train)

# perform prediction on x_test data
# predicts labels on test set
y_pred = classifier.predict(x_test)
#test_set['prediction']=y_pred
#print(y_pred)

# creating confusion matrix and accuracy calculation
cm = confusion_matrix(y_test,y_pred) # shows what model got right and wrong
print(cm)
accuracy = float(cm.diagonal().sum())/len(y_test) # calculates correct predictions as a percentage
print('model accuracy is:',accuracy*100,'%')


#x1_test = [[73,6]] # for new data testing
