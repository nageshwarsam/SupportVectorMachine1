# import libraries for data handling, clustering, model training, and evaluation
import pandas as pd
import numpy as np
from sklearn.datasets import make_blobs                  # to generate fake dataset
from sklearn.preprocessing import LabelEncoder, StandardScaler  # for label conversion + scaling
from sklearn.model_selection import train_test_split     # for train/test split
from sklearn.cluster import KMeans                       # for unsupervised clustering
from sklearn.svm import SVC                              # for support vector machine
from sklearn.metrics import confusion_matrix             # to evaluate model

# step 1: generate fake data w/ 4 features + 3 clusters
# simulates real ocean sensor data grouped into 3 types of impact
X, _ = make_blobs(n_samples=300, centers=3, n_features=4, random_state=42)

# step 2: use kmeans to find 3 clusters in the data
# unsupervised = no label needed yet
kmeans = KMeans(n_clusters=3, random_state=42)
cluster_labels = kmeans.fit_predict(X)  # outputs 0, 1, or 2 for each point

# step 3: turn cluster numbers into readable labels (just for show)
# you can change this order based on what each cluster really means
label_mapping = {0: "Low", 1: "Medium", 2: "High"}
mapped_labels = [label_mapping[label] for label in cluster_labels]

# step 4: encode labels to numbers so svm can use them (low=0, med=1, high=2)
le = LabelEncoder()
y = le.fit_transform(mapped_labels)

# step 5: split data into 80% train and 20% test
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# step 6: normalize features (make them 0 mean, 1 std dev)
# helps rbf kernel work better, avoids bias due to scale
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# step 7: train svm on the training data
# rbf = non-linear kernel, ovo = one-vs-one (for multi-class)
classifier = SVC(kernel='rbf', C=1, gamma='auto', decision_function_shape='ovo', random_state=1)
classifier.fit(x_train, y_train)  # learn from train data

# step 8: make predictions using test data
y_pred = classifier.predict(x_test)

# step 9: check performance
# confusion matrix shows counts of correct/incorrect for each class
cm = confusion_matrix(y_test, y_pred)

# accuracy = (# correct) / (total)
accuracy = float(cm.diagonal().sum()) / len(y_test)

# show confusion matrix w/ label names
import ace_tools as tools; tools.display_dataframe_to_user(
    name="Confusion Matrix", 
    dataframe=pd.DataFrame(cm, 
        index=[f"True {label}" for label in le.classes_], 
        columns=[f"Pred {label}" for label in le.classes_])
)

# print accuracy %
print(f'model accuracy is: {accuracy * 100:.2f}%')
