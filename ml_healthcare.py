# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import time

# %%
# Load the dataset
path = './sample_data/breast_cancer.csv'
data = pd.read_csv(path, header=None)

# %%
# assign the column names
data.columns = data.iloc[0]

# continue with the next rows
# data = data[1:]

# %%
# Count the number of benign and malignant cases
class_counts = data['Class'].value_counts()
benign_count = class_counts[2]
malignant_count = class_counts[4]

# Create a pie chart
labels = ['Benign', 'Malignant']
sizes = [benign_count, malignant_count]
colors = ['skyblue', 'lightcoral']
explode = (0, 0.1)  # Explode the second slice

plt.figure(figsize=(8, 6))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
plt.title('Distribution of Class')
plt.show()

# %%
import seaborn as sns
# Select the "Clump Thickness" column
clump_thickness = data['Clump_Thickness']

# Create the histogram
plt.figure(figsize=(8, 6))
sns.histplot(clump_thickness, kde=True, color='blue')

# Add labels and title
plt.xlabel('Clump Thickness')
plt.ylabel('Frequency')
plt.title('Distribution of Clump Thickness')

# Display the histogram
plt.show()

# %%
from sklearn.neighbors import KNeighborsClassifier

# use first row as column names
data.columns = data.iloc[0]
data = data[1:]

# Replace '?' with NaN
data.replace('?', np.nan, inplace=True)

# Convert the DataFrame to numeric values
data = data.astype(float)

# Split the data into features and target variable
X = data.drop('Class', axis=1)
y = data['Class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Impute missing values with the mean
numeric_imputer = SimpleImputer(strategy='mean')
X_train = numeric_imputer.fit_transform(X_train)
X_test = numeric_imputer.transform(X_test)

# Train the KNN classifier
knn = KNeighborsClassifier(n_neighbors=3)

start_time = time.time()

knn.fit(X_train, y_train)

training_time = time.time() - start_time
start_time = time.time()

# make the predictions
y_pred = knn.predict(X_test)

testing_time = time.time() - start_time

# calculate accuracy
knn_accuracy = accuracy_score(y_test, y_pred)
print("KNN Accuracy:", knn_accuracy)
print("KNN Training Time:", training_time)
print("KNN Testing Time:", testing_time)

# %%
from sklearn.ensemble import RandomForestClassifier

# Impute missing values with the mean
numeric_imputer = SimpleImputer(strategy='mean')
X_train = numeric_imputer.fit_transform(X_train)
X_test = numeric_imputer.transform(X_test)

# create and train the RF classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)

start_time = time.time()

rf.fit(X_train, y_train)

training_time = time.time() - start_time

# make the predictions
start_time = time.time()
y_pred = rf.predict(X_test)
testing_time = time.time() - start_time

# calculate accuracy
rfc_accuracy = accuracy_score(y_test, y_pred)
print("RFC Accuracy:", rfc_accuracy)
print("RFC Training Time:", training_time)
print("RFC Testing Time:", testing_time)

# %%
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

# Drop the missing values (rows with '?')
data = data.replace('?', pd.NA).dropna()

# Split the data into features (X) and target (y)
X = data.drop('Class', axis=1)
y = data['Class']

# Convert the target variable to numerical labels
le = LabelEncoder()
y = le.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the SVM classifier
svm = SVC()
start_time = time.time()
svm.fit(X_train, y_train)
training_time = time.time() - start_time

# Make the predictions
start_time = time.time()
y_pred = svm.predict(X_test)
testing_time = time.time() - start_time

# calculate accuracy
svc_accuracy = accuracy_score(y_test, y_pred)
print("SVC Accuracy:", svc_accuracy)
print("SVC Training Time:", training_time)
print("SVC Testing Time:", testing_time)

# %%
import xgboost as xgb
# Create an XGBoost classifier
xgb_classifier = xgb.XGBClassifier()

# Train the XGBoost classifier
start_time = time.time()
xgb_classifier.fit(X_train, y_train)
training_time = time.time() - start_time

# Make predictions on the test set
start_time = time.time()
y_pred = xgb_classifier.predict(X_test)
testing_time = time.time() - start_time

# calculate accuracy
xgb_accuracy = accuracy_score(y_test, y_pred)
print("XGB Accuracy:", xgb_accuracy)
print("XGB Training Time:", training_time)
print("XGB Testing Time:", testing_time)

# %%
# Create a bar plot
algorithms = ['KNN', 'RF', 'SVM', 'XGB']
accuracies = [knn_accuracy, rfc_accuracy, svc_accuracy, xgb_accuracy]

plt.bar(algorithms, accuracies)
plt.xlabel('Algorithms')
plt.ylabel('Accuracy')
plt.title('Performance Comparison')
plt.ylim(0.5, 1.0)  # Set the y-axis limits
plt.show()


