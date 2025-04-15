import pandas as pd
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

# Step 1: Generate synthetic data (to simulate ocean device features)
# We simulate 3 clusters to represent low, medium, and high environmental impact
X, _ = make_blobs(n_samples=300, centers=3, n_features=4, random_state=42)

# Step 2: Perform KMeans clustering (unsupervised) to simulate labeling
# This mimics the clustering stage in the paper
kmeans = KMeans(n_clusters=3, random_state=42)
cluster_labels = kmeans.fit_predict(X)

# Step 3: Map numeric cluster labels to human-readable classes (optional for realism)
# You can remap this based on domain knowledge (e.g., 0=Low, 1=Med, 2=High)
label_mapping = {0: "Low", 1: "Medium", 2: "High"}
mapped_labels = [label_mapping[label] for label in cluster_labels]

# Step 4: Encode labels for SVM (SVM requires numeric labels)
le = LabelEncoder()
y = le.fit_transform(mapped_labels)

# Step 5: Split into training and test datasets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Step 6: Feature scaling (important for SVMs with RBF or polynomial kernels)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Step 7: Train an SVM classifier
# 'rbf' is a non-linear kernel; ovo = one-vs-one for multi-class support
classifier = SVC(kernel='rbf', C=1, gamma='auto', decision_function_shape='ovo', random_state=1)
classifier.fit(x_train, y_train)

# Step 8: Predict on the test set
y_pred = classifier.predict(x_test)

# Step 9: Evaluate using a confusion matrix and accuracy score
cm = confusion_matrix(y_test, y_pred)
accuracy = float(cm.diagonal().sum()) / len(y_test)

# Output the results
import ace_tools as tools; tools.display_dataframe_to_user(name="Confusion Matrix", dataframe=pd.DataFrame(cm, 
    index=[f"True {label}" for label in le.classes_], 
    columns=[f"Pred {label}" for label in le.classes_]))

print(f'Model accuracy is: {accuracy * 100:.2f}%')
